"""Template loader and renderer."""

import uuid
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.datagen.sql2ibis.template_loader.expander import expand_template_variations


class Template:
    """SQL+Ibis template with variations."""

    def __init__(self, data: Dict[str, Any]):
        self.name = data["name"]
        self.description = data.get("description", "")
        self.difficulty = data.get("difficulty", "medium")
        self.features = data.get("features", [])
        self.sql_template = data["sql_template"]
        self.ibis_template = data["ibis_template"]
        self.variations = data.get("variations", [])
        self.context = data.get("context", {})
        self.parameter_space = data.get("parameter_space", None)

        # Expand variations with parameter space if present
        # Check for template-level OR variation-level parameter spaces
        needs_expansion = self.parameter_space or any(
            "parameter_space" in v for v in self.variations
        )
        if needs_expansion:
            self.variations = self._expand_variations()

    def _expand_variations(self) -> List[Dict[str, Any]]:
        """Expand variations using parameter space."""
        expanded = []

        # If template-level parameter_space, expand all variations
        if self.parameter_space:
            for base_variation in self.variations:
                expanded_vars = expand_template_variations(
                    base_variation, self.parameter_space, name_pattern="{base_name}_{idx}"
                )
                expanded.extend(expanded_vars)
        else:
            # Check each variation for parameter_space
            for base_variation in self.variations:
                var_param_space = base_variation.get("parameter_space")

                if var_param_space:
                    # Expand this variation
                    expanded_vars = expand_template_variations(
                        base_variation, var_param_space, name_pattern="{base_name}_{idx}"
                    )
                    expanded.extend(expanded_vars)
                else:
                    # No expansion, use as-is
                    expanded.append(base_variation)

        return expanded

    def render(self, variation: Dict[str, Any]) -> Dict[str, Any]:
        """Render a specific variation of this template.

        Parameters
        ----------
        variation : dict
            Variation parameters

        Returns
        -------
        dict
            Rendered example with SQL, Ibis code, and metadata
        """
        params = variation.get("params", {})

        # Render SQL
        sql = self.sql_template.format(**params).strip()

        # Render Ibis code
        ibis_code = self.ibis_template.format(**params).strip()

        # Create example
        example = {
            "id": str(uuid.uuid4()),
            "task": "sql_to_ibis",
            "dialect": "duckdb",
            "backend": "duckdb",
            "ibis_version": "9.5.0",  # Will be dynamically set later
            "context": self.context,
            "input": {"sql": sql},
            "target": {
                "ibis": ibis_code,
                "expr_name": "expr",
            },
            "meta": {
                "template": self.name,
                "variation": variation.get("name", "default"),
                "features": self.features,
                "source": "synthetic",
                "difficulty": self.difficulty,
            },
        }

        return example


def load_templates(template_dir: Path) -> List[Template]:
    """Load all templates from directory.

    Parameters
    ----------
    template_dir : Path
        Directory containing YAML templates

    Returns
    -------
    list of Template
        Loaded templates
    """
    templates = []

    for yaml_file in sorted(template_dir.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
            templates.append(Template(data))

    return templates


def generate_examples(templates: List[Template]) -> List[Dict[str, Any]]:
    """Generate all examples from templates.

    Parameters
    ----------
    templates : list of Template
        Templates to render

    Returns
    -------
    list of dict
        Generated examples
    """
    examples = []

    for template in templates:
        for variation in template.variations:
            example = template.render(variation)
            examples.append(example)

    return examples
