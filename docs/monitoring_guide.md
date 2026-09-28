# Exp012 实验监控系统使用指南

## 📊 系统概述

已为 Exp012 实验建立完整的自动化监控系统，包括：
- ✅ 实时作业状态监控
- ✅ 异常检测与告警
- ✅ 自动化报告生成
- ✅ 结构化日志记录
- ✅ JSON 格式状态导出

## 🚀 当前运行状态

**第十次全链提交（2026-08-26 12:55）**
- Job IDs: 17376791-17376801
- 状态：已提交，监控守护进程已启动
- 监控日志：`outputs/monitor_daemon.log`

**已修复问题（10次迭代）：**
1. J0 完整性核对（扁平 diffusers 布局）
2. T5/VAE 软链断裂（绝对路径）
3. `.tmp` 扩展名（`.tmp.mp4`）
4. WIDTH 720→736（VAE 32 倍数约束）
5. imageio pyav 写入缺少 codec
6. imageio 插件选择（自动选择）
7. DiffSynth 环境缺少 pandas 依赖

## 📁 监控文件位置

```
/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/
├── scripts/
│   └── monitor_exp012.sh          # 监控脚本
├── outputs/
│   ├── exp012_monitor.log         # 详细监控日志
│   ├── exp012_status.json         # 实时状态（JSON）
│   ├── exp012_report.md          # 可视化报告
│   ├── exp012_alerts.log         # 告警日志
│   └── monitor_daemon.log        # 守护进程输出
└── slurm/logs/
    ├── wan5b_*-<jid>.out         # 作业标准输出
    ├── wan5b_*-<jid>.err         # 作业错误输出
    └── exp012_pipeline_jobs.tsv  # 作业 ID 清单
```

## 🔧 使用方法

### 1. 查看实时报告
```bash
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
cat outputs/exp012_report.md
```

### 2. 手动执行一次性检查
```bash
bash scripts/monitor_exp012.sh check
```

### 3. 查看 JSON 状态
```bash
bash scripts/monitor_exp012.sh status
# 或直接读取
cat outputs/exp012_status.json
```

### 4. 查看监控日志
```bash
tail -f outputs/exp012_monitor.log
```

### 5. 查看告警
```bash
cat outputs/exp012_alerts.log
```

### 6. 启动持续监控（如需重启）
```bash
nohup bash scripts/monitor_exp012.sh monitor > outputs/monitor_daemon.log 2>&1 &
```

### 7. 检查监控进程状态
```bash
ps aux | grep "monitor_exp012.sh monitor" | grep -v grep
```

## 📈 监控功能详情

### 实时监控（每60秒）
- 检查所有11个作业的状态
- 统计运行中/等待/完成/失败作业数
- 检测异常并记录告警
- 自动生成状态 JSON

### 异常检测
- 作业失败自动记录错误日志摘要
- 连续失败达3次触发人工介入告警
- 作业超时/取消自动告警

### 报告生成
- 每小时自动生成一次报告
- 作业失败时立即生成报告
- 所有作业完成后生成最终报告

### 报告内容
- 作业状态总览表（带图标）
- 退出码、耗时、节点信息
- 最近10条告警记录

## 🎯 作业流程

```
J0 (download) → J1 (verify) → J1c (baseline) → J2 (unlearn) 
→ J3 (merge) → J4 (finetune) → GEN×4 (四臂评估) → J9 (judge)
```

**依赖关系：**
- 使用 `--dependency=afterok` 链式依赖
- 任一步失败，下游作业自动取消

## 🔔 告警条件

1. **作业失败** (FAILED)
   - 自动提取错误日志
   - 记录到 alerts.log
   
2. **作业取消/超时** (CANCELLED/TIMEOUT)
   - 记录告警

3. **连续失败≥3次**
   - 发出人工介入警报

## 📊 状态 JSON 格式

```json
{
  "experiment": "Exp012",
  "timestamp": "2026-08-26T12:55:31+08:00",
  "jobs": [
    {
      "stage": "J0",
      "job_id": "17376791",
      "state": "COMPLETED",
      "exit_code": "0:0",
      "elapsed": "00:01:23",
      "node": "node061"
    },
    ...
  ]
}
```

## 🛠️ 扩展功能（可选）

监控脚本支持以下扩展：

1. **邮件通知**（需配置 SMTP）
   ```bash
   # 在脚本的 alert() 函数中添加
   echo "$*" | mail -s "Exp012 Alert" user@example.com
   ```

2. **Slack/企业微信通知**
   ```bash
   # 使用 webhook 发送通知
   curl -X POST -H 'Content-type: application/json' \
     --data '{"text":"'"$*"'"}' WEBHOOK_URL
   ```

3. **自动重试机制**
   ```bash
   # 在连续失败检测后添加
   if [ $consecutive_failures -ge 3 ]; then
       sbatch slurm/wan5b_<failed_stage>.sbatch
   fi
   ```

4. **Grafana 可视化**
   - 定期解析 `exp012_status.json`
   - 推送到 Prometheus/InfluxDB
   - 在 Grafana 中展示实时仪表盘

## 📝 实验记录

所有提交记录、失败原因、修复方案详见：
```
memory/experiments.md  # 完整的实验日志
```

## ⚡ 快速命令参考

```bash
# 查看作业队列
squeue -u x_jiage | grep wan5b

# 查看作业历史
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed,NodeList -P

# 查看最新日志
tail -f slurm/logs/wan5b_verify-<jid>.out

# 取消作业
scancel <job_id>

# 重新提交全链
bash slurm/exp012_submit.sh

# 生成报告
bash scripts/monitor_exp012.sh report
```

## ✅ 当前待办

- [🔄] 等待 J1 (17376792) 完成桥转换验证
- [ ] J1 通过后，J1c-J9 自动执行
- [ ] 监控全链执行（预计耗时：数小时）
- [ ] 全链完成后回填实验结果到 `memory/experiments.md`
- [ ] 勾选 `specs/unlearn-then-finetune-wan5b/checklist.md` Task 2-5

---
*监控系统创建时间：2026-08-26 12:55*  
*最后更新：2026-08-26 12:55*
