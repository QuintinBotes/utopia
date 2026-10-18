//! 定时推理的入队：库在两拍之间一直到期，排着的那份任务不能再多一份。

use sqlx::PgPool;
use uuid::Uuid;

#[tokio::test]
async fn a_due_base_is_queued_once_while_its_job_waits() -> anyhow::Result<()> {
    let Some(url) = crate::test_db::url() else {
        return Ok(());
    };
    let pool = PgPool::connect(&url).await?;
    let (org, ws, kb) = (Uuid::now_v7(), Uuid::now_v7(), Uuid::now_v7());
    sqlx::raw_sql(&format!(
        "INSERT INTO organizations(id,name) VALUES ('{org}','inference-schedule');
         INSERT INTO workspaces(id,org_id,name) VALUES ('{ws}','{org}','inference-schedule');
         INSERT INTO knowledge_bases(id,workspace_id,name,materialize_inferences)
              VALUES ('{kb}','{ws}','inference-schedule',true);"
    ))
    .execute(&pool)
    .await?;

    // 两拍之间没人认领：从没推过的库两拍都算到期
    super::queue_due_inferences(&pool).await?;
    super::queue_due_inferences(&pool).await?;
    let queued: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM jobs
          WHERE kind = 'materialize_inferences' AND status = 'queued' AND payload->>'kb_id' = $1",
    )
    .bind(kb.to_string())
    .fetch_one(&pool)
    .await?;

    sqlx::query("DELETE FROM jobs WHERE payload->>'kb_id' = $1")
        .bind(kb.to_string())
        .execute(&pool)
        .await?;
    sqlx::query("DELETE FROM organizations WHERE id = $1")
        .bind(org)
        .execute(&pool)
        .await?;
    assert_eq!(queued, 1);
    Ok(())
}
