"""
记忆模块 - 长期记忆与回忆系统（支持持久化）
"""

import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional


class Memory:
    """Agent 的记忆系统（支持 SQLite 持久化）"""

    def __init__(self, capacity: int = 1000, db_path: str = None, auto_save: bool = True):
        self.capacity = capacity
        self.short_term: List[Dict] = []  # 短期记忆（内存缓存）
        self.long_term: List[Dict] = []  # 长期记忆（内存缓存）
        self.semantic: Dict[str, any] = {}  # 语义记忆（知识）
        self.episodic: List[Dict] = []  # 情景记忆（经历）

        # 线程锁
        self._lock = threading.RLock()

        # 持久化配置
        self.auto_save = auto_save
        self.db = None

        # 初始化数据库
        if db_path is not None or self._should_use_db():
            try:
                from core.memory_db import MemoryDatabase

                self.db = MemoryDatabase(db_path)
                self._load_from_db()
                print(f"✅ 记忆数据库已加载：{self.db.db_path}")
            except Exception as e:
                print(f"⚠️ 记忆数据库初始化失败：{e}，将使用内存模式")
                self.db = None

    def add_short_term(self, content: str, category: str = "general", importance: float = 0.5):
        """添加短期记忆"""
        # 输入验证
        if not content or not isinstance(content, str):
            raise ValueError("Content must be a non-empty string")
        if not isinstance(category, str):
            raise ValueError("Category must be a string")
        if not 0.0 <= importance <= 1.0:
            raise ValueError("Importance must be between 0.0 and 1.0")

        with self._lock:
            memory = {
                "id": self._generate_id(content),
                "content": content,
                "category": category,
                "importance": importance,
                "created_at": datetime.now().isoformat(),
                "access_count": 0,
                "last_accessed": None,
            }
            self.short_term.append(memory)

            # 自动保存到数据库
            if self.db and self.auto_save:
                self.db.save_short_term(memory)

            # 超出容量时清理
            if len(self.short_term) > self.capacity:
                self._consolidate_memories()

            return memory["id"]

    def _generate_id(self, content: str) -> str:
        """生成记忆 ID"""
        # 使用UUID生成唯一ID，更安全且不易碰撞
        return str(uuid.uuid4())[:12]

    def _should_use_db(self) -> bool:
        """检查是否应该使用数据库"""
        # 只要 db_path 为 None（使用默认路径）或明确指定了路径，就使用数据库
        return True

    def _load_from_db(self):
        """从数据库加载记忆"""
        if not self.db:
            return

        try:
            # 加载短期记忆
            stm_data = self.db.get_all_short_term(limit=self.capacity)
            self.short_term = stm_data

            # 加载长期记忆
            ltm_data = self.db.get_all_long_term(limit=self.capacity)
            self.long_term = ltm_data

            # 加载语义记忆
            self.semantic = self.db.get_all_semantic()

            # 加载情景记忆
            self.episodic = self.db.get_all_episodic(limit=500)

            print(
                f"📚 已加载记忆：短期 {len(self.short_term)} 条，长期 {len(self.long_term)} 条，语义 {len(self.semantic)} 条，情景 {len(self.episodic)} 条"
            )
        except Exception as e:
            print(f"⚠️ 加载记忆失败：{e}")
            # 确保数据结构初始化
            self.short_term = []
            self.long_term = []
            self.semantic = {}
            self.episodic = []

    def get_all_short_term(self, limit: int = 1000):
        """获取所有短期记忆"""
        return self.short_term[:limit]

    def get_all_long_term(self, limit: int = 1000):
        """获取所有长期记忆"""
        return self.long_term[:limit]

    def add_long_term(self, content: str, category: str = "general", tags: List[str] = None):
        """添加长期记忆"""
        with self._lock:
            memory = {
                "id": self._generate_id(content),
                "content": content,
                "category": category,
                "tags": tags or [],
                "created_at": datetime.now().isoformat(),
                "access_count": 0,
            }
            self.long_term.append(memory)

            if self.db and self.auto_save:
                self.db.save_long_term(memory)

            return memory["id"]

    def _save_to_db(self):
        """保存记忆到数据库"""
        if not self.db or not self.auto_save:
            return

        try:
            # 保存短期记忆
            for memory in self.short_term[-10:]:  # 只保存最近的
                self.db.save_short_term(memory)

            # 保存长期记忆
            for memory in self.long_term[-10:]:
                self.db.save_long_term(memory)

            # 保存语义记忆
            for key, value in self.semantic.items():
                self.db.save_semantic(key, value)

            # 情景记忆在添加时已保存

            # 提交数据库事务
            if self.db and hasattr(self.db, "conn") and self.db.conn:
                self.db.conn.commit()
        except Exception as e:
            print(f"⚠️ 保存记忆失败：{e}")

    def _consolidate_memories(self):
        """记忆巩固 - 将重要的短期记忆转为长期记忆"""
        with self._lock:
            # 按重要性排序
            self.short_term.sort(key=lambda x: x["importance"], reverse=True)

            # 转移重要的到长期记忆
            to_transfer = self.short_term[: int(len(self.short_term) * 0.2)]
            for memory in to_transfer:
                memory["consolidated_at"] = datetime.now().isoformat()
                self.long_term.append(memory)
                self.short_term.remove(memory)

            # 删除最不重要的
            if len(self.short_term) > self.capacity * 0.8:
                self.short_term = self.short_term[: int(self.capacity * 0.8)]

    def access_memory(self, memory_id: str) -> Optional[Dict]:
        """访问记忆"""
        with self._lock:
            for memory in self.short_term + self.long_term:
                if memory["id"] == memory_id:
                    memory["access_count"] += 1
                    memory["last_accessed"] = datetime.now().isoformat()
                    return memory
            return None

    def search_memories(self, keywords: str, limit: int = 10) -> List[Dict]:
        """搜索记忆"""
        with self._lock:
            # 预处理关键词
            keywords_lower = keywords.lower()
            keyword_list = keywords_lower.split()

            # 搜索结果和分数
            results = []

            # 搜索短期和长期记忆
            for memory in self.short_term + self.long_term:
                memory_content = memory["content"].lower()

                # 计算匹配分数
                score = 0

                # 完全匹配
                if keywords_lower in memory_content:
                    score += 2.0

                # 关键词匹配
                for keyword in keyword_list:
                    if keyword in memory_content:
                        score += 0.5

                # 重要性和访问次数加权
                score += memory.get("importance", 0.5) * 0.3
                score += memory.get("access_count", 0) * 0.1

                if score > 0:
                    results.append((score, memory))

            # 按分数排序
            results.sort(key=lambda x: x[0], reverse=True)

            # 提取前limit个结果
            return [memory for _, memory in results[:limit]]

    def add_semantic(self, key: str, value: any):
        """添加语义记忆（知识）"""
        with self._lock:
            self.semantic[key] = value

    def add_episodic(self, event: str, context: Dict = None):
        """添加情景记忆（经历）"""
        episodic_memory = {"event": event, "context": context or {}, "timestamp": datetime.now().isoformat()}

        with self._lock:
            self.episodic.append(episodic_memory)

            # 自动保存到数据库
            if self.db and self.auto_save:
                self.db.save_episodic(event, context)

            if len(self.episodic) > 500:
                self.episodic = self.episodic[-500:]

    def get_memory_summary(self) -> str:
        """获取记忆摘要"""
        with self._lock:
            return """
【记忆系统】
短期记忆：{len(self.short_term)}/{self.capacity}
长期记忆：{len(self.long_term)}
语义知识：{len(self.semantic)} 条
情景记忆：{len(self.episodic)} 条

【最近记忆】
{chr(10).join(f"• {m['content'][:50]}..." for m in self.short_term[-5:]) if self.short_term else '无'}

【最常访问】
{chr(10).join(f"• {m['content'][:50]}... ({m['access_count']}次)" for m in sorted(self.long_term, key=lambda x: x['access_count'], reverse=True)[:3]) if self.long_term else '无'}
"""

    def to_dict(self) -> Dict:
        """序列化"""
        with self._lock:
            return {
                "short_term": self.short_term[-50:],
                "long_term": self.long_term[-100:],
                "semantic": self.semantic,
                "episodic": self.episodic[-100:],
            }

    def save_all(self):
        """强制保存所有记忆到数据库"""
        if not self.db:
            print("⚠️ 数据库未初始化，无法保存")
            return

        try:
            with self._lock:
                # 保存短期记忆
                stm_count = 0
                for memory in self.short_term:
                    try:
                        self.db.save_short_term(memory)
                        stm_count += 1
                    except Exception as e:
                        print(f"⚠️ 保存短期记忆失败：{e}")

                # 保存长期记忆
                ltm_count = 0
                for memory in self.long_term:
                    try:
                        self.db.save_long_term(memory)
                        ltm_count += 1
                    except Exception as e:
                        print(f"⚠️ 保存长期记忆失败：{e}")

                # 保存语义记忆
                semantic_count = 0
                for key, value in self.semantic.items():
                    try:
                        self.db.save_semantic(key, value)
                        semantic_count += 1
                    except Exception as e:
                        print(f"⚠️ 保存语义记忆失败：{e}")

                # 保存元数据
                try:
                    self.db.save_metadata("last_saved", datetime.now().isoformat())
                    self.db.save_metadata("stats", self.db.get_stats())
                except Exception as e:
                    print(f"⚠️ 保存元数据失败：{e}")

                print(f"💾 记忆已保存到数据库：{self.db.db_path}")
                print(
                    f"   统计：短期 {stm_count}/{len(self.short_term)}，长期 {ltm_count}/{len(self.long_term)}，语义 {semantic_count}/{len(self.semantic)}"
                )
        except Exception as e:
            print(f"❌ 保存记忆失败：{e}")

    def close(self):
        """关闭数据库连接"""
        if self.db:
            self.save_all()  # 关闭前自动保存
            self.db.close()
            print("✅ 记忆数据库已关闭")
