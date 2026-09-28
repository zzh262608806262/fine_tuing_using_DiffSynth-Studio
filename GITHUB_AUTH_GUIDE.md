# GitHub 认证配置指南

## 问题
推送失败：`Permission denied to nanthanmd-coder`

## 原因
当前 git 配置使用了错误的 GitHub 账户或没有认证凭据。

## 解决方案

### 方案 1: 使用 Personal Access Token (推荐)

1. **生成 GitHub Personal Access Token**
   - 访问: https://github.com/settings/tokens
   - 点击 "Generate new token (classic)"
   - 勾选权限: `repo` (完整仓库访问权限)
   - 生成并复制 token（只显示一次！）

2. **配置 Git 使用 Token**
   ```bash
   cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
   
   # 方法 A: 在 URL 中包含 token
   git remote set-url origin https://YOUR_TOKEN@github.com/zzh262608806262/fine_tuing_using_DiffSynth-Studio.git
   
   # 方法 B: 使用 git credential helper (更安全)
   git config --global credential.helper store
   # 下次 push 时输入用户名和 token，会自动保存
   ```

3. **推送**
   ```bash
   git push origin migration-cleanup
   ```

### 方案 2: 使用 SSH Key

1. **生成 SSH Key**
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   # 按回车使用默认路径
   
   # 查看公钥
   cat ~/.ssh/id_ed25519.pub
   ```

2. **添加到 GitHub**
   - 访问: https://github.com/settings/keys
   - 点击 "New SSH key"
   - 粘贴公钥内容

3. **修改远程 URL 为 SSH**
   ```bash
   git remote set-url origin git@github.com:zzh262608806262/fine_tuing_using_DiffSynth-Studio.git
   ```

4. **推送**
   ```bash
   git push origin migration-cleanup
   ```

### 方案 3: 在浏览器中手动推送

如果上述方法都不行，可以：

1. 创建 GitHub 仓库的 ZIP 包
   ```bash
   cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
   git archive -o /tmp/migration-cleanup.zip migration-cleanup
   ```

2. 在 GitHub 网页界面上传文件

## 当前 Git 配置检查

```bash
# 检查当前用户配置
git config --global user.name
git config --global user.email

# 检查远程仓库 URL
git remote -v

# 检查是否有保存的凭据
git config --global credential.helper
```

## 正确配置应该是

```bash
# 设置正确的用户信息
git config --global user.name "zzh262608806262"  # 或你的 GitHub 用户名
git config --global user.email "your_email@example.com"

# 使用 HTTPS + token 或 SSH
```

---

**下一步**: 选择一种方案配置认证，然后重新推送。
