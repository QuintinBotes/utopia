//! 限额就地调整（#1302）：调高放进已经在等的任务；调低要等在飞的调用还回来才生效。

use super::ModelGates;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

const SOON: Duration = Duration::from_millis(100);

#[tokio::test]
async fn raising_the_limit_admits_a_task_already_waiting() {
    let gates = Arc::new(ModelGates::default());
    let _in_flight = gates.admit("m", 1).await.unwrap();
    let g = gates.clone();
    let mut waiting = tokio::spawn(async move { g.admit("m", 1).await });
    assert!(timeout(SOON, &mut waiting).await.is_err());

    gates.resize("m", 2);
    assert!(timeout(SOON, &mut waiting).await.is_ok());
}

#[tokio::test]
async fn lowering_the_limit_takes_effect_as_in_flight_calls_return() {
    let gates = Arc::new(ModelGates::default());
    let first = gates.admit("m", 2).await.unwrap();
    let second = gates.admit("m", 2).await.unwrap();
    let g = gates.clone();
    let mut waiting = tokio::spawn(async move { g.admit("m", 1).await });

    // 两个在飞降到一个：还回来的第一张被收走，第二个调用还没完就仍满额
    drop(first);
    assert!(timeout(SOON, &mut waiting).await.is_err());
    drop(second);
    assert!(timeout(SOON, &mut waiting).await.is_ok());
}

#[tokio::test]
async fn raising_back_before_a_lowering_settles_keeps_the_higher_limit() {
    let gates = ModelGates::default();
    let first = gates.admit("m", 2).await.unwrap();
    let _second = gates.admit("m", 2).await.unwrap();
    gates.resize("m", 1);
    gates.resize("m", 2);

    // 欠账被调高抵掉了：一张还回来，限额 2 下就能再进一个
    drop(first);
    assert!(timeout(SOON, gates.admit("m", 2)).await.is_ok());
}
