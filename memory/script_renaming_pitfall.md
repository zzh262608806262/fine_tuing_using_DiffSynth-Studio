# 常见陷阱：脚本重命名导致的失败

## 问题描述
- **现象**：作业启动后立即失败（<1分钟），提示找不到 Python 脚本
- **根因**：项目演进过程中脚本被重命名，但旧的 SLURM 配置/文档还引用旧名称

## 具体案例

### 案例1: unlearn_wan5b.py → run_unlearn_wan5b.py
- **时间**：2026-09-03 Exp014 失败
- **错误**：`can't open file 'scripts/unlearn_wan5b.py': No such file or directory`
- **原因**：早期脚本叫 `unlearn_wan5b.py`，后改名为 `run_unlearn_wan5b.py`
- **影响**：Exp012（早期）能跑通，Exp014（复制旧配置）失败
- **修复**：将所有引用更新为 `run_unlearn_wan5b.py`

## 预防措施

1. **提交作业后立即检查前5分钟**
   - 脚本路径错误通常在启动时就失败
   - 用 `sacct -j <job_id>` 检查 Elapsed 时间，<1分钟失败很可能是路径问题

2. **重命名脚本时同步更新所有引用**
   - SLURM 脚本（`slurm/*.sbatch`）
   - 文档（`docs/*.md`, `memory/*.md`）
   - 其他脚本中的调用

3. **使用绝对路径或相对路径检查**
   ```bash
   # 在 sbatch 脚本中先检查文件存在
   SCRIPT="scripts/run_unlearn_wan5b.py"
   if [ ! -f "$SCRIPT" ]; then
       echo "错误: 脚本不存在 $SCRIPT"
       exit 1
   fi
   python "$SCRIPT" ...
   ```

4. **记录脚本重命名历史**
   - 在 `memory/errors.md` 记录重命名事件
   - 标注受影响的实验编号

## 相关脚本名称变更记录

| 旧名称 | 新名称 | 变更时间 | 受影响实验 |
|--------|--------|----------|-----------|
| `unlearn_wan5b.py` | `run_unlearn_wan5b.py` | ~2026-08 | Exp014 |
| （待补充） | （待补充） | - | - |

## 检查清单

当复制旧实验配置时：
- [ ] 检查所有脚本路径是否存在：`ls scripts/<script_name>.py`
- [ ] 检查参数名称是否匹配：`python scripts/<script>.py --help`
- [ ] 先 dry-run 或提交 1分钟测试作业验证路径正确
