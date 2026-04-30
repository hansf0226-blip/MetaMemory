# MetaMemory Makefile v2.2
# 用法: make [target]

.PHONY: help install install-dev install-full test test-unit test-integration \
        benchmark benchmark-full bench-quick coverage lint format clean \
        docker-build docker-up docker-down

# ── 默认目标 ──────────────────────────────────────────────────
help:
	@echo "MetaMemory Makefile v2.2"
	@echo ""
	@echo "安装:"
	@echo "  install           安装核心依赖"
	@echo "  install-dev       安装开发依赖"
	@echo "  install-full      安装全部依赖（含 ML）"
	@echo ""
	@echo "测试:"
	@echo "  test              运行全部测试"
	@echo "  test-unit         只运行单元测试"
	@echo "  test-integration  只运行集成测试"
	@echo ""
	@echo "基准测试:"
	@echo "  benchmark         标准 benchmark (500 docs)"
	@echo "  bench-full        完整 benchmark (2000 docs)"
	@echo "  bench-quick       快速 benchmark (10 docs)"
	@echo ""
	@echo "代码质量:"
	@echo "  coverage          生成覆盖率报告"
	@echo "  lint              代码检查"
	@echo "  format            代码格式化"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build      构建 Docker 镜像"
	@echo "  docker-up         启动 Docker 服务"
	@echo "  docker-down       停止 Docker 服务"
	@echo ""
	@echo "其他:"
	@echo "  clean             清理临时文件"

# ── 安装 ──────────────────────────────────────────────────────

install:
	pip install -e "."

install-dev:
	pip install -e ".[dev]"
	pre-commit install 2>/dev/null || true

install-full:
	pip install -e ".[all]"

install-semantic:
	pip install -e ".[semantic]"

# ── 测试 ──────────────────────────────────────────────────────

test:
	@echo "🧪 Running all tests..."
	python -m pytest tests/ -v --tb=short

test-unit:
	@echo "🧪 Running unit tests..."
	python -m pytest tests/test_core_yijing.py tests/test_i_ching.py -v --tb=short

test-integration:
	@echo "🧪 Running integration tests..."
	python -m pytest tests/ -m "integration" -v --tb=short 2>/dev/null || \
	python -m pytest tests/test_lifecycle_scheduler.py tests/test_hybrid_retrieval.py tests/test_identity_policy.py -v --tb=short

# ── 基准测试 ──────────────────────────────────────────────────

benchmark:
	@echo "📊 Running standard benchmark..."
	python -m core.benchmarks

bench-full:
	@echo "📊 Running full benchmark (2000 docs)..."
	python -m core.benchmarks --full

bench-quick:
	@echo "📊 Running quick benchmark (50 docs)..."
	python -m core.benchmarks --quick

bench-json: benchmark
	@echo "📄 Saving to benchmark_report.json..."
	python -m core.benchmarks --output benchmark_report.json

# ── 覆盖率 ────────────────────────────────────────────────────

coverage:
	@echo "📊 Generating coverage report..."
	mkdir -p reports
	python -m pytest tests/ \
		--cov=core \
		--cov-report=term-missing \
		--cov-report=html:reports/coverage \
		--cov-report=xml:reports/coverage.xml
	@echo "✅ Reports: reports/coverage/index.html"

# ── 代码质量 ──────────────────────────────────────────────────

lint:
	@echo "🔍 black..."
	python -m black --check core/ tests/ 2>/dev/null || true
	@echo "🔍 isort..."
	python -m isort --check core/ tests/ 2>/dev/null || true

format:
	@echo "✨ Formatting..."
	python -m black core/ tests/
	python -m isort core/ tests/

# ── Docker ────────────────────────────────────────────────────

docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t metamemory:v2.2 .

docker-up:
	@echo "🐳 Starting services..."
	docker-compose up -d

docker-down:
	@echo "🐳 Stopping services..."
	docker-compose down

docker-bench:
	@echo "🐳 Running benchmark in Docker..."
	docker run --rm metamemory:v2.2 python -m core.benchmarks --quick

# ── 清理 ──────────────────────────────────────────────────────

clean:
	@echo "🧹 Cleaning..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage reports/ htmlcov/ .mypy_cache/
	rm -rf dist/ build/
	rm -f *.db data/*.db
	@echo "✅ Cleaned"

# ── 提交前检查 ────────────────────────────────────────────────

pre-commit: format test
	@echo "✅ Pre-commit checks passed"
