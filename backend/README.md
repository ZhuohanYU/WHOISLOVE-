# backend/ — 本地测试数据管理

## 目录结构

```
backend/
  seed.py          恢复：fixtures → 数据库
  dump.py          保存：数据库 → fixtures
  fixtures/
    user.json      你的用户资料
    user_files.json  你上传的文件记录（元数据）
    targets.json   所有对象 + 人格分析 + 约会记录
```

## 常用命令

```bash
# 进入虚拟环境
source .venv/bin/activate

# 首次或重置测试环境（清空数据库后从 fixtures 恢复）
python backend/seed.py --clean

# 追加到现有数据库（不清空）
python backend/seed.py

# 把当前数据库状态保存到 fixtures（下次可恢复）
python backend/dump.py
```

## 工作流程

1. 开始一次测试 → `python backend/seed.py --clean`
2. 在网页里测试，填数据、跑模拟
3. 测完觉得数据有价值 → `python backend/dump.py` 保存
4. 下次继续 → `python backend/seed.py`（不加 --clean 则追加）

## 注意

- `fixtures/` 里的 JSON 文件可以直接手动编辑，修改测试数据
- 上传的文件（图片/PDF）不会被 dump 复制，只保留元数据记录
- `targets.json` 里每个对象包含 `_personality`（人格分析）和 `_sessions`（所有约会记录）
