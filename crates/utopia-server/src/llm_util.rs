//! 从工作区设置构造 LLM 客户端，以及按模型的并发闸门。

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use utopia_core::models::LlmSettings;
use utopia_llm::LlmClient;

use crate::state::AppState;

pub fn chat_client(s: &LlmSettings) -> Option<LlmClient> {
    if !s.chat_ready() {
        return None;
    }
    Some(
        LlmClient::new(
            s.chat_base_url.as_deref()?,
            s.chat_api_key.as_deref(),
            s.chat_model.as_deref()?,
        )
        .with_reasoning_effort(s.chat_reasoning_effort.clone()),
    )
}

/// 对话模型的设置：地址、密钥、模型、推理力度。任一项变了就是另一个客户端
type ChatConfig = (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
);

/// 各工作区的对话客户端，跨轮次留着。
///
/// 端点报过的上下文窗口记在客户端上（`LlmClient::remember_context_window`），每轮新建一个
/// 就等于每轮都忘，下一轮照样撞上同一个 400（#964）。模型设置一变就换新的：学到的窗口
/// 是那个模型的，不是这个工作区的。密钥和窗口都不跨工作区
#[derive(Default)]
pub struct ConversationClients {
    inner: std::sync::Mutex<HashMap<uuid::Uuid, (ChatConfig, LlmClient)>>,
}

impl ConversationClients {
    pub fn get(&self, workspace: uuid::Uuid, s: &LlmSettings) -> Option<LlmClient> {
        let mut clients = self.inner.lock().unwrap();
        if !s.chat_ready() {
            clients.remove(&workspace);
            return None;
        }
        let config = (
            s.chat_base_url.clone(),
            s.chat_api_key.clone(),
            s.chat_model.clone(),
            s.chat_reasoning_effort.clone(),
        );
        if let Some((previous, client)) = clients.get(&workspace) {
            if previous == &config {
                return Some(client.clone());
            }
        }
        let client = chat_client(s)?;
        clients.insert(workspace, (config, client.clone()));
        Some(client)
    }
}

/// 一轮对话里嵌过的文字与它的向量（#971）。挑口径、检索分块、按名字查实体都要把问题嵌成
/// 向量，一轮之内是同一句话、同一个模型，嵌一次就够。缓存跟着这一轮的 `ToolSink` 走，不落库，
/// 只记成功的；没有这一轮的调用方（HTTP 检索、口径页）传 `None`，各嵌各的
pub type EmbedCache = std::collections::HashMap<String, Vec<f32>>;

pub fn embed_client(s: &LlmSettings) -> Option<LlmClient> {
    if !s.embed_ready() {
        return None;
    }
    Some(LlmClient::new(
        s.embed_base_url.as_deref()?,
        s.embed_api_key.as_deref(),
        s.embed_model.as_deref()?,
    ))
}

/// 按模型的信号量注册表：每个模型一把，限额变了就地调整（#1302）。
///
/// 原来限额一变就换一把新的。等在旧的那把上的任务不会挪过去，每个都要拿旧限额再做
/// 完一次调用才换到新的上——抽取一次调用几分钟，八个 worker 排在一张许可后面，限额
/// 从 1 调到 4 之后好几个小时还是一次只跑一个，看起来就是「改了不生效」。
///
/// 调高用 `add_permits`，已经在等的立刻放进来。调低时空着的许可当场收走
/// （`forget_permits`）；正被在飞调用占着的记成欠账，调用还回来时由下一个取许可的
/// 收掉。没用「另起一个任务把差额 acquire 下来再 forget」：那个任务拿到之前若又调高，
/// 它会把新加的许可一起吞掉，限额就停在低的那一档；欠账在调高时先抵消，没这个问题。
#[derive(Default)]
pub struct ModelGates {
    inner: std::sync::Mutex<HashMap<String, Gate>>,
}

struct Gate {
    sem: Arc<Semaphore>,
    limit: usize,
    /// 调低之后还该收走、当时正被在飞调用占着的许可数
    owed: usize,
}

impl ModelGates {
    async fn admit(&self, key: &str, limit: usize) -> Option<OwnedSemaphorePermit> {
        let sem = self.resize(key, limit);
        loop {
            let permit = sem.clone().acquire_owned().await.ok()?;
            if !self.repay(key) {
                return Some(permit);
            }
            permit.forget();
        }
    }

    fn resize(&self, key: &str, limit: usize) -> Arc<Semaphore> {
        let mut gates = self.inner.lock().unwrap();
        let gate = gates.entry(key.to_string()).or_insert_with(|| Gate {
            sem: Arc::new(Semaphore::new(limit)),
            limit,
            owed: 0,
        });
        if limit > gate.limit {
            let raise = limit - gate.limit;
            let cancelled = raise.min(gate.owed);
            gate.owed -= cancelled;
            gate.sem.add_permits(raise - cancelled);
        } else if limit < gate.limit {
            let lower = gate.limit - limit;
            gate.owed += lower - gate.sem.forget_permits(lower);
        }
        gate.limit = limit;
        gate.sem.clone()
    }

    /// 还欠着许可就该收掉刚拿到的这张
    fn repay(&self, key: &str) -> bool {
        match self.inner.lock().unwrap().get_mut(key) {
            Some(gate) if gate.owed > 0 => {
                gate.owed -= 1;
                true
            }
            _ => false,
        }
    }
}

/// 后台任务调模型前取一张许可，持有到调用结束。
///
/// **只给后台任务用**（抽取、裁决、摄入嵌入、本体建议）。用户的对话与检索
/// 不走这里——让人打字等在十个后台抽取后面，产品就成了坏的；真正会打爆
/// 供应商速率限制的也从来不是一个人在打字。
///
/// 限额读不到（表还没建、库暂时不可达）时**放行**：并发限制是保护措施，
/// 不该因为读不到配置而把整条流水线卡死。
pub async fn acquire(
    state: &AppState,
    base_url: &str,
    model: &str,
) -> Option<OwnedSemaphorePermit> {
    let limit = utopia_store::model_limits::limit_for(&state.pool, base_url, model)
        .await
        .ok()?;
    let key = format!("{base_url}|{model}");
    state.model_gates.admit(&key, limit).await
}

/// `acquire` 的便捷形式：直接从工作区设置取 chat 模型的身份。
pub async fn acquire_chat(state: &AppState, s: &LlmSettings) -> Option<OwnedSemaphorePermit> {
    let (base, model) = (s.chat_base_url.as_deref()?, s.chat_model.as_deref()?);
    acquire(state, base, model).await
}

/// `acquire` 的便捷形式：embedding 模型。
pub async fn acquire_embed(state: &AppState, s: &LlmSettings) -> Option<OwnedSemaphorePermit> {
    let (base, model) = (s.embed_base_url.as_deref()?, s.embed_model.as_deref()?);
    acquire(state, base, model).await
}

#[cfg(test)]
#[path = "llm_util_tests.rs"]
mod tests;
