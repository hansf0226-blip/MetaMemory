"""
RAG 检索引擎 - 检索增强生成
融合向量检索与 LLM 生成，提供知识增强的回答
"""

import json
from datetime import datetime
from typing import Dict, List

try:
    from api.llm_manager import LLMManager
except ImportError:
    LLMManager = None
from core.embedding import EmbeddingGenerator
from core.vector_store import VectorDatabase


class RAGEngine:
    """RAG 检索引擎 - 检索增强生成"""

    def __init__(
        self,
        vector_db: VectorDatabase = None,
        embedding_gen: EmbeddingGenerator = None,
        llm: LLMManager = None,
        db_path: str = "data/vector_store.db",
    ):
        # 初始化组件
        self.vector_db = vector_db or VectorDatabase(db_path=db_path)
        self.embedding_gen = embedding_gen or EmbeddingGenerator(llm)
        self.llm = llm or LLMManager()

        # RAG 配置
        self.config = {
            "top_k": 5,  # 检索文档数
            "min_similarity": 0.3,  # 最小相似度
            "max_context_length": 2000,  # 最大上下文长度
            "use_rerank": True,  # 是否重排序
            "include_metadata": True,  # 包含元数据
            "cache_results": True,  # 缓存结果
        }

        # 检索历史
        self.search_history: List[Dict] = []
        self.total_searches = 0

        # 结果缓存
        self.result_cache: Dict[str, List[Dict]] = {}

        print("✅ RAG 检索引擎已初始化")
        print(f"   检索文档数：{self.config['top_k']}")
        print(f"   最小相似度：{self.config['min_similarity']}")
        print(f"   最大上下文：{self.config['max_context_length']}")
        print(f"   重排序：{'✅' if self.config['use_rerank'] else '❌'}")

    # ========== 核心检索 ==========

    def search(
        self, query: str, top_k: int = None, tags_filter: List[str] = None, include_knowledge: bool = True
    ) -> Dict:
        """
        语义检索

        Args:
            query: 查询文本
            top_k: 返回数量
            tags_filter: 标签过滤
            include_knowledge: 是否包含知识库检索

        Returns:
            检索结果（包含文档和元数据）
        """
        top_k = top_k or self.config["top_k"]

        # 检查缓存
        cache_key = f"{query}:{top_k}:{json.dumps(tags_filter)}"
        if self.config["cache_results"] and cache_key in self.result_cache:
            # 缓存中存储的是完整结果字典
            return self.result_cache[cache_key]

        print(f"🔍 检索：{query[:50]}...")

        # 生成查询嵌入
        query_embedding = self.embedding_gen.generate_embedding(query)

        # 检索向量
        vector_results = self.vector_db.search_similar(
            query_embedding, top_k=top_k, tags_filter=tags_filter, min_similarity=self.config["min_similarity"]
        )

        # 检索知识库
        knowledge_results = []
        if include_knowledge:
            knowledge_results = self.vector_db.search_knowledge(query_embedding, top_k=top_k, min_importance=0.3)

        # 合并结果
        results = self._merge_results(vector_results, knowledge_results, top_k)

        # 重排序
        if self.config["use_rerank"] and results:
            results = self._rerank_results(results, query)

        # 记录历史
        self._record_search(query, results)

        # 缓存结果
        if self.config["cache_results"]:
            self.result_cache[cache_key] = {
                "query": query,
                "results": results,
                "total_found": len(results),
                "timestamp": datetime.now().isoformat(),
            }

        print(f"   ✅ 找到 {len(results)} 个相关文档")

        return {
            "query": query,
            "results": results,
            "total_found": len(results),
            "timestamp": datetime.now().isoformat(),
        }

    def _merge_results(self, vector_results: List[Dict], knowledge_results: List[Dict], top_k: int) -> List[Dict]:
        """合并检索结果"""
        all_results = []

        # 添加向量结果
        for r in vector_results:
            all_results.append(
                {
                    "type": "vector",
                    "content": r["content"],
                    "similarity": r["similarity"],
                    "metadata": r.get("metadata", {}),
                    "tags": r.get("tags", []),
                    "source": "memory",
                }
            )

        # 添加知识库结果
        for r in knowledge_results:
            all_results.append(
                {
                    "type": "knowledge",
                    "title": r["title"],
                    "content": r["content"],
                    "similarity": r["similarity"],
                    "category": r.get("category"),
                    "importance": r.get("importance", 0.5),
                    "source_url": r.get("source_url"),
                    "source": "knowledge_base",
                }
            )

        # 按相似度排序
        all_results.sort(key=lambda x: x["similarity"], reverse=True)

        return all_results[:top_k]

    def _rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """使用 LLM 重排序结果"""
        if not self.llm or not self.llm.providers:
            return results

        # 简单实现：基于内容相关性评分
        query_lower = query.lower()

        for result in results:
            content = result.get("content", "").lower()

            # 计算关键词重叠
            query_words = set(query_lower.split())
            content_words = set(content.split())

            overlap = len(query_words & content_words) / max(len(query_words), 1)

            # 调整相似度分数
            result["rerank_score"] = result["similarity"] * 0.7 + overlap * 0.3

        # 按重排序分数排序
        results.sort(key=lambda x: x.get("rerank_score", x["similarity"]), reverse=True)

        return results

    def _record_search(self, query: str, results: List[Dict]):
        """记录检索历史"""
        self.search_history.append(
            {
                "query": query,
                "results_count": len(results),
                "avg_similarity": sum(r["similarity"] for r in results) / len(results) if results else 0,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.total_searches += 1

        # 限制历史记录大小
        if len(self.search_history) > 100:
            self.search_history = self.search_history[-100:]

    # ========== RAG 增强生成 ==========

    def generate_with_context(
        self, query: str, system_prompt: str = None, top_k: int = None, include_sources: bool = True
    ) -> Dict:
        """
        使用检索上下文生成回答

        Args:
            query: 用户查询
            system_prompt: 系统提示
            top_k: 检索文档数
            include_sources: 是否包含来源

        Returns:
            生成的回答（包含上下文和来源）
        """
        # 检索相关文档
        search_result = self.search(query, top_k=top_k)
        results = search_result["results"]

        # 构建上下文
        context = self._build_context(results)

        # 构建提示词
        prompt = self._build_rag_prompt(query, context, system_prompt)

        # 调用 LLM 生成
        if self.llm and self.llm.providers:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.chat(messages, temperature=0.7, max_tokens=1000)
        else:
            response = "[模拟模式] 基于检索到的知识，我来回答您的问题..."

        # 构建返回
        result = {
            "query": query,
            "response": response,
            "context_used": context,
            "sources": results if include_sources else [],
            "search_stats": (
                {
                    "total_searches": self.total_searches,
                    "results_found": len(results),
                    "avg_similarity": search_result["results"][0]["similarity"] if results else 0,
                }
                if results
                else None
            ),
            "timestamp": datetime.now().isoformat(),
        }

        return result

    def _build_context(self, results: List[Dict]) -> str:
        """构建上下文"""
        if not results:
            return "未找到相关知识。"

        context_parts = []
        total_length = 0
        max_length = self.config["max_context_length"]

        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            source = result.get("source", "unknown")

            part = f"[{i}] [{source}] {content}"

            if total_length + len(part) > max_length:
                break

            context_parts.append(part)
            total_length += len(part)

        return "\n\n".join(context_parts)

    def _build_rag_prompt(self, query: str, context: str, system_prompt: str = None) -> str:
        """构建 RAG 提示词"""
        default_system = """你是一个知识丰富的 AI 助手。请基于以下检索到的知识来回答用户的问题。

要求：
1. 优先使用检索到的知识
2. 如果知识不足，可以补充你的通用知识
3. 回答要准确、清晰、有条理
4. 如果检索到的知识与问题不相关，请说明"""

        system = system_prompt or default_system

        prompt = """{system}

【检索到的知识】
{context}

【用户问题】
{query}

请回答："""

        return prompt

    # ========== 知识管理 ==========

    def add_knowledge(
        self,
        title: str,
        content: str,
        category: str = None,
        tags: List[str] = None,
        importance: float = 0.5,
        source_url: str = None,
    ) -> str:
        """添加知识到知识库"""
        # 生成嵌入
        embedding = self.embedding_gen.generate_embedding(content)

        # 添加到向量库
        vector_id = self.vector_db.add_vector(
            content=content, embedding=embedding, metadata={"title": title, "category": category}, tags=tags
        )

        # 添加到知识库
        kb_id = self.vector_db.add_knowledge(
            title=title,
            content=content,
            embedding=embedding,
            category=category,
            metadata={"tags": tags},
            source_url=source_url,
            importance=importance,
        )

        print(f"✅ 添加知识：{title} ({category})")

        return kb_id

    def add_knowledge_batch(self, items: List[Dict]) -> int:
        """批量添加知识"""
        print(f"📚 批量添加 {len(items)} 条知识...")

        # 生成所有嵌入
        contents = [item.get("content", "") for item in items]
        embeddings = self.embedding_gen.generate_embeddings_batch(contents)

        # 添加到向量库
        vector_items = []
        for i, item in enumerate(items):
            vector_items.append(
                {
                    "content": item.get("content", ""),
                    "embedding": embeddings[i],
                    "metadata": {"title": item.get("title", ""), "category": item.get("category")},
                    "tags": item.get("tags", []),
                }
            )

        self.vector_db.add_vectors_batch(vector_items)

        # 添加到知识库
        kb_items = []
        for i, item in enumerate(items):
            kb_items.append(
                {
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "embedding": embeddings[i],
                    "category": item.get("category"),
                    "metadata": {"tags": item.get("tags", [])},
                    "source_url": item.get("source_url"),
                    "importance": item.get("importance", 0.5),
                }
            )

        self.vector_db.add_knowledge_batch(kb_items)

        return len(items)

    # ========== 统计与状态 ==========

    def get_stats(self) -> Dict:
        """获取统计信息"""
        vector_stats = self.vector_db.get_stats()
        kb_stats = self.vector_db.get_knowledge_stats()
        embedding_stats = self.embedding_gen.get_cache_stats()

        return {
            "total_searches": self.total_searches,
            "vector_count": vector_stats["vector_count"],
            "knowledge_count": vector_stats["knowledge_count"],
            "knowledge_categories": kb_stats["categories"],
            "embedding_cache_hit_rate": embedding_stats["hit_rate"],
            "cache_size": embedding_stats["cache_size"],
        }

    def print_status(self):
        """打印状态"""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("【RAG 检索引擎状态】")
        print("=" * 60)
        print(f"总检索次数：{stats['total_searches']}")
        print(f"向量总数：{stats['vector_count']}")
        print(f"知识库条目：{stats['knowledge_count']}")
        print(f"知识分类：{stats['knowledge_categories']}")
        print(f"嵌入缓存命中率：{stats['embedding_cache_hit_rate']*100:.1f}%")
        print(f"嵌入缓存大小：{stats['cache_size']}")
        print("=" * 60)

    def print_search_history(self, limit: int = 10):
        """打印检索历史"""
        print("\n【最近检索】")

        for i, record in enumerate(self.search_history[-limit:], 1):
            time_str = record["timestamp"].split("T")[1].split(".")[0]
            print(
                f"  {i}. [{time_str}] \"{record['query'][:40]}...\" "
                + f"(找到{record['results_count']}个，相似度：{record['avg_similarity']:.2f})"
            )
