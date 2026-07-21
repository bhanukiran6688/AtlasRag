import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.services.query_planner import QueryPlanner


class TestQueryPlanner:
    @pytest.fixture
    def mock_gateway(self):
        gateway = Mock()
        gateway.generate = AsyncMock()
        return gateway

    @pytest.fixture
    def query_planner(self, mock_gateway):
        return QueryPlanner(mock_gateway)

    @pytest.mark.asyncio
    async def test_build_queries_returns_original_when_disabled(
        self, query_planner, monkeypatch
    ):
        monkeypatch.setattr(
            "src.config.settings.settings",
            Mock(
                query_planning_adaptive_enabled=False,
                cost_optimization_max_query_planning_calls=1,
                query_expansion_max_variants=2,
                query_decomposition_max_subquestions=2,
            ),
        )

        result = await query_planner.build_queries(
            "test question", use_query_expansion=False, use_query_decomposition=False
        )
        assert result == ["test question"]

    @pytest.mark.asyncio
    async def test_build_queries_with_expansion_enabled(self, query_planner, monkeypatch):
        monkeypatch.setattr(
            "src.config.settings.settings",
            Mock(
                query_planning_adaptive_enabled=False,
                cost_optimization_max_query_planning_calls=1,
                query_expansion_max_variants=2,
                query_decomposition_max_subquestions=2,
                query_planning_min_words_for_expansion=5,
            ),
        )

        result = await query_planner.build_queries(
            "test question with many words",
            use_query_expansion=True,
            use_query_decomposition=False,
        )
        assert len(result) >= 1
        assert "test question with many words" in result

    @pytest.mark.asyncio
    async def test_build_queries_with_decomposition_enabled(self, query_planner, monkeypatch):
        monkeypatch.setattr(
            "src.config.settings.settings",
            Mock(
                query_planning_adaptive_enabled=False,
                cost_optimization_max_query_planning_calls=1,
                query_expansion_max_variants=2,
                query_decomposition_max_subquestions=2,
                query_planning_min_words_for_decomposition=5,
            ),
        )

        result = await query_planner.build_queries(
            "test question with many words",
            use_query_expansion=False,
            use_query_decomposition=True,
        )
        assert len(result) >= 1

    def test_should_expand_returns_true_when_adaptive_disabled(
        self, query_planner, monkeypatch
    ):
        with patch("src.services.query_planner.settings.query_planning_adaptive_enabled", False):
            result = query_planner._should_expand("test", None)
            assert result is True

    def test_should_expand_with_low_confidence(self, query_planner, monkeypatch):
        monkeypatch.setattr(
            "src.config.settings.settings",
            Mock(
                query_planning_adaptive_enabled=True,
                query_planning_low_confidence_threshold=0.5,
                query_planning_min_words_for_expansion=10,
            ),
        )

        result = query_planner._should_expand("test", 0.3)
        assert result is True

    def test_should_expand_with_high_confidence_and_short_question(
        self, query_planner, monkeypatch
    ):
        monkeypatch.setattr(
            "src.config.settings.settings",
            Mock(
                query_planning_adaptive_enabled=True,
                query_planning_low_confidence_threshold=0.5,
                query_planning_min_words_for_expansion=10,
            ),
        )

        result = query_planner._should_expand("test", 0.8)
        assert result is False

    def test_should_decompose_with_multi_part_question(
        self, query_planner, monkeypatch
    ):
        monkeypatch.setattr(
            "src.config.settings.settings",
            Mock(
                query_planning_adaptive_enabled=True,
                query_planning_low_confidence_threshold=0.5,
                query_planning_min_words_for_decomposition=10,
            ),
        )

        result = query_planner._should_decompose("compare A and B", None)
        assert result is True

    def test_is_low_confidence_returns_true_when_below_threshold(self, monkeypatch):
        monkeypatch.setattr(
            "src.config.settings.settings",
            Mock(query_planning_low_confidence_threshold=0.5),
        )

        result = QueryPlanner._is_low_confidence(0.3)
        assert result is True

    def test_is_low_confidence_returns_false_when_above_threshold(self, monkeypatch):
        monkeypatch.setattr(
            "src.config.settings.settings",
            Mock(query_planning_low_confidence_threshold=0.5),
        )

        result = QueryPlanner._is_low_confidence(0.8)
        assert result is False

    def test_is_low_confidence_returns_false_when_none(self, monkeypatch):
        monkeypatch.setattr(
            "src.config.settings.settings",
            Mock(query_planning_low_confidence_threshold=0.5),
        )

        result = QueryPlanner._is_low_confidence(None)
        assert result is False

    def test_heuristic_variants_generates_alternatives(
        self, query_planner, monkeypatch
    ):
        monkeypatch.setattr(
            "src.config.settings.settings", Mock(query_expansion_max_variants=3)
        )

        result = query_planner._heuristic_variants("test query")
        assert len(result) > 0
        assert "test query" in result

    def test_heuristic_variants_with_empty_string(self, query_planner):
        result = query_planner._heuristic_variants("")
        assert result == []

    def test_heuristic_subquestions_splits_on_conjunctions(
        self, query_planner, monkeypatch
    ):
        monkeypatch.setattr(
            "src.config.settings.settings", Mock(query_decomposition_max_subquestions=3)
        )

        result = query_planner._heuristic_subquestions("What is A and what is B")
        assert len(result) >= 1

    def test_heuristic_subquestions_with_simple_question(
        self, query_planner, monkeypatch
    ):
        monkeypatch.setattr(
            "src.config.settings.settings", Mock(query_decomposition_max_subquestions=3)
        )

        result = query_planner._heuristic_subquestions("What is test")
        assert len(result) >= 1

    def test_extract_keywords_removes_stop_words(self):
        result = QueryPlanner._extract_keywords("what is the test")
        assert "what" not in result
        assert "is" not in result
        assert "the" not in result
        assert "test" in result

    def test_extract_keywords_limits_to_top_keywords(self):
        result = QueryPlanner._extract_keywords("word1 word2 word3 word4 word5 word6")
        assert len(result) <= 4

    def test_deduplicate_removes_duplicates(self):
        result = QueryPlanner._deduplicate(["query1", "query1", "query2", "QUERY2"])
        assert len(result) == 2
        assert "query1" in result
        assert "query2" in result

    def test_deduplicate_normalizes_whitespace(self):
        result = QueryPlanner._deduplicate(["query  1", "query 1", "query2"])
        assert len(result) == 2

    def test_extract_string_list_with_valid_json(self):
        result = QueryPlanner._extract_string_list(
            '{"queries": ["q1", "q2"]}', "queries", 5
        )
        assert result == ["q1", "q2"]

    def test_extract_string_list_with_invalid_json(self):
        result = QueryPlanner._extract_string_list("invalid json", "queries", 5)
        assert result == []

    def test_extract_string_list_with_missing_key(self):
        result = QueryPlanner._extract_string_list('{"other": ["q1"]}', "queries", 5)
        assert result == []

    def test_extract_string_list_respects_limit(self):
        result = QueryPlanner._extract_string_list(
            '{"queries": ["q1", "q2", "q3", "q4"]}', "queries", 2
        )
        assert len(result) == 2

    def test_extract_string_list_strips_whitespace(self):
        result = QueryPlanner._extract_string_list(
            '{"queries": [" q1 ", " q2 "]}', "queries", 5
        )
        assert result == ["q1", "q2"]

    @pytest.mark.asyncio
    async def test_generate_variants_falls_back_on_error(
        self, query_planner, mock_gateway, monkeypatch
    ):
        monkeypatch.setattr(
            "src.config.settings.settings", Mock(query_expansion_max_variants=2)
        )

        mock_gateway.generate.side_effect = Exception("API error")
        result = await query_planner._generate_variants("test query")
        assert result == []

    @pytest.mark.asyncio
    async def test_generate_subquestions_falls_back_on_error(
        self, query_planner, mock_gateway, monkeypatch
    ):
        monkeypatch.setattr(
            "src.config.settings.settings", Mock(query_decomposition_max_subquestions=2)
        )

        mock_gateway.generate.side_effect = Exception("API error")
        result = await query_planner._generate_subquestions("test query")
        assert result == []
