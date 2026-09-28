# Exp020a 执行清单

**创建日期**: 2026-09-16  
**最后更新**: 2026-09-16 18:12  
**状态**: Phase 2 - 完整流程执行中

---

## ✅ Phase 1: Smoke Test - 已完成

### 执行记录
- ❌ Job 17536581 - 失败（diffusers未安装）
- ✅ Job 17536586 - 成功完成

### 验证结果
```yaml
mask_config:
  attn_substring: "attn2"
  text_len: 0
  is_joint: false
  num_heads: 48        # ✅ 验证结果
  frames: 4            # ✅ 验证结果
  height: 30           # ✅ 验证结果
  width: 46            # ✅ 验证结果
```

---

## 🔄 Phase 2: 完整流程执行中

### 作业提交时间
2026-09-16 18:12

### Job依赖链
```
17536588 (擦除训练) - 预计30分钟
  └─> 17536589 (合并) - 预计3分钟
        └─> 17536590 (生成) - 预计2小时
              └─> 17536591 (评估) - 预计30分钟
```

### 预计完成时间
- 擦除训练: ~18:42
- 模型合并: ~18:45
- 视频生成: ~20:45
- 安全评估: ~21:15

**总计**: 约3小时（预计21:15完成）

---

## 📊 监控命令

查看作业状态：
```bash
squeue -u x_jiage | grep exp020a
```

查看日志：
```bash
# 擦除训练
tail -f slurm/logs/exp020a_unlearn-17536588.out

# 合并
tail -f slurm/logs/exp020a_merge-17536589.out

# 生成
tail -f slurm/logs/exp020a_generate-17536590.out

# 评估
tail -f slurm/logs/exp020a_evaluate-17536591.out
```

---

**当前状态**: 所有作业已提交，等待完成
**下一步**: 等待评估完成后分析结果
