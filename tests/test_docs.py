import dataclasses
import unittest

from gravsphere2 import config

from .helpers import REPO


class TestDocumentation(unittest.TestCase):
    def test_every_config_key_is_documented(self):
        text = (REPO / "docs" / "input-file.md").read_text()
        missing = []
        for cls in (config.Config, config.OutputConfig, config.DataConfig, config.PhotometryConfig,
                    config.GridConfig, config.StellarMassConfig, config.TracerPriorConfig,
                    config.AnisotropyPriorConfig, config.MassPriorConfig, config.SamplerConfig):
            for f in dataclasses.fields(cls):
                if f.metadata.get("yaml", True) and f"`{f.name}`" not in text and f"{f.name}" not in text:
                    missing.append(f"{cls.__name__}.{f.name}")
        self.assertEqual(missing, [])

    def test_template_lists_every_key_and_loads(self):
        template = REPO / "examples" / "template_full.yaml"
        cfg = config.Config.load(template)
        text = template.read_text()
        for cls in (config.OutputConfig, config.DataConfig, config.GridConfig, config.StellarMassConfig,
                    config.SamplerConfig, config.MassPriorConfig, config.AnisotropyPriorConfig):
            for f in dataclasses.fields(cls):
                self.assertIn(f.name, text, f"{cls.__name__}.{f.name} missing from template")
        self.assertEqual(cfg.sampler.nlive, 500)

    def test_mkdocs_nav_files_exist(self):
        import re

        nav = (REPO / "mkdocs.yml").read_text()
        for name in re.findall(r":\s+([\w\-]+\.md)", nav):
            self.assertTrue((REPO / "docs" / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
