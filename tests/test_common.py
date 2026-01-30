"""Tests for DSAT common utilities."""

import pytest


class TestSQLTemplates:
    """Test SQL Template Engine."""

    def test_import(self):
        """Test that SQLTemplateEngine can be imported."""
        from dsat.common import SQLTemplateEngine
        assert SQLTemplateEngine is not None

    def test_validate_column_name(self):
        """Test column name validation."""
        from dsat.common import validate_column_name
        
        assert validate_column_name("age") == True
        assert validate_column_name("column_name") == True
        assert validate_column_name("123invalid") == False
        assert validate_column_name("has space") == False

    def test_sanitize_column_name(self):
        """Test column name sanitization."""
        from dsat.common import sanitize_column_name
        
        assert sanitize_column_name("has space") == "has_space"
        assert sanitize_column_name("123start") == "_123start"


class TestDAGGenerator:
    """Test DAG Generator."""

    def test_import(self):
        """Test that DAGGenerator can be imported."""
        from dsat.common import DAGGenerator
        assert DAGGenerator is not None

    def test_initialization(self):
        """Test DAGGenerator initialization."""
        from dsat.common import DAGGenerator
        
        generator = DAGGenerator(
            project_id="test-project",
            dataset_id="test_dataset",
            source_table="test_table",
            target_dataset="transformed"
        )
        
        assert generator.project_id == "test-project"
        assert generator.dataset_id == "test_dataset"
        assert generator.source_table == "test_table"
        assert generator.target_table == "test_table_transformed"

    def test_generate(self):
        """Test DAG generation."""
        from dsat.common import DAGGenerator
        
        generator = DAGGenerator(
            project_id="test-project",
            dataset_id="test_dataset", 
            source_table="test_table",
            target_dataset="transformed"
        )
        
        transformations = [
            {"column_name": "age", "fe_method": "standardization"},
            {"column_name": "salary", "fe_method": "median_imputation"},
        ]
        
        result = generator.generate(transformations, target_column="target")
        
        assert result["status"] == "success"
        assert "dag_code" in result
        assert "CREATE OR REPLACE TABLE" in result["dag_code"]
