# 贡献指南

感谢您对 OpenTrade 项目的兴趣！我们欢迎所有形式的社区贡献。

## 🤝 贡献方式

- 🐛 **报告 Bug**: 发现问题请提交 Issue，详细描述复现步骤和环境信息
- 💡 **功能建议**: 提出新功能想法，帮助项目变得更好
- 📝 **文档改进**: 完善官方文档、翻译、示例代码
- 🔧 **代码贡献**: 提交 Pull Request 修复 Bug 或新增功能
- 🎁 **策略分享**: 将您的自定义策略插件分享到社区
- 🌐 **社区维护**: 帮助解答社区问题、维护交流群

## 🚀 开始贡献

### 1. Fork 仓库

```bash
# 访问 https://github.com/opentrade-ai/opentrade
# 点击右上角 Fork 按钮

git clone https://github.com/YOUR_USERNAME/opentrade.git
cd opentrade
```

### 2. 创建开发分支

```bash
git checkout -b feature/your-amazing-feature
```

### 3. 设置开发环境

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -e ".[dev]"

# 安装 pre-commit 钩子
pre-commit install
```

### 4. 运行测试

```bash
# 确保所有测试通过
pytest tests/ -v

# 运行代码质量检查
ruff check opentrade/
mypy opentrade/
black --check opentrade/
```

### 5. 提交代码

```bash
# 遵循 Conventional Commits 规范
git add .
git commit -m "feat: 新增功能描述"
git push origin feature/your-amazing-feature
```

### 6. 创建 Pull Request

访问 https://github.com/opentrade-ai/opentrade/pulls
点击 "New Pull Request"

## 📋 代码规范

### Python

- 遵循 [PEP 8](https://pep8.org/) 规范
- 使用 Black 格式化代码
- 使用 isort 排序导入
- 使用 mypy 进行类型检查
- 提交信息遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范

### Git 提交规范

```
feat: 新功能
fix: Bug 修复
docs: 文档更新
style: 代码格式调整
refactor: 重构
perf: 性能优化
test: 测试相关
chore: 构建/工具相关
```

示例:
- `feat: 新增趋势跟踪策略`
- `fix: 修复交易所连接超时问题`
- `docs: 更新快速开始指南`

### 测试要求

- 核心代码测试覆盖率不低于 **80%**
- 新功能必须包含对应的单元测试
- 所有测试必须通过

## 🏗️ 项目结构

```
opentrade/
├── opentrade/           # 核心包
│   ├── agents/         # AI Agents
│   ├── services/       # 业务服务
│   ├── models/         # 数据模型
│   ├── plugins/        # 插件系统
│   ├── cli/           # 命令行
│   └── core/          # 核心配置
├── tests/              # 测试
├── docs/               # 文档
└── scripts/            # 脚本
```

## 💬 社区

- 📧 邮箱: contributors@opentrade.ai
- 💬 Discord: https://discord.gg/opentrade
- 🐦 Twitter: https://twitter.com/opentrade_ai

## ⚠️ 重要提示

提交代码即表示您同意将代码以 MIT 许可证开源。

---

**感谢您的贡献！** 🎉
