"""Retrieval tests for the LLaVA prompt context. No model weights required."""

import unittest

from rag_context import build_glaucoma_prompt, build_screen_prompt, retrieve


DIABETES_PROMPT = """[INST] <image>
Examine this fundus photograph for diabetic retinopathy.
[/INST]"""

RVO_PROMPT = """[INST] <image>
Examine this fundus photograph for retinal vein occlusion.
[/INST]"""

GLAUCOMA_PROMPT = """[INST] <image>
Describe the cup-to-disc ratio and any rim thinning.
[/INST]"""


class RetrieveTests(unittest.TestCase):
    def test_microaneurysms_retrieve_diabetes_signs(self):
        hits = retrieve("microaneurysms hard exudates")
        self.assertEqual(hits[0]["id"], "diabetes.signs")
        self.assertNotIn("branch rvo", hits[0]["text"].lower())

    def test_geographic_atrophy_retrieves_amd(self):
        hits = retrieve("geographic atrophy")
        self.assertEqual(hits[0]["id"], "macular_degeneration.signs")

    def test_condition_filter_drops_other_diseases(self):
        hits = retrieve("haemorrhage", condition="rvo")
        ids = {hit["id"] for hit in hits}
        self.assertIn("rvo.signs", ids)
        self.assertNotIn("diabetes.signs", ids)


class ScreenPromptTests(unittest.TestCase):
    def test_diabetes_prompt_keeps_image_token_and_allowed_grades(self):
        built = build_screen_prompt("diabetes", DIABETES_PROMPT)
        prompt = built["prompt"]
        self.assertTrue(prompt.startswith("[INST] <image>"))
        self.assertIn("Proliferative Diabetic Retinopathy (PDR)=6", prompt)
        self.assertIn("No Suspect or Negative=2", prompt)
        self.assertNotIn("Branch RVO", prompt)
        ids = [hit["id"] for hit in built["retrieved"]]
        self.assertIn("diabetes.grades", ids)
        self.assertIn("diabetes.signs", ids)

    def test_rvo_prompt_does_not_import_diabetes_grades(self):
        built = build_screen_prompt("rvo", RVO_PROMPT)
        self.assertIn("Suspect or Positive=25", built["prompt"])
        self.assertIn("No Suspect or Negative=24", built["prompt"])
        self.assertNotIn("Proliferative Diabetic Retinopathy", built["prompt"])

    def test_existing_and_other_eye_are_retrieved(self):
        built = build_screen_prompt(
            "diabetes",
            DIABETES_PROMPT,
            existing_ads=4,
            other_eye_ads=6,
            eye="left",
        )
        prompt = built["prompt"]
        ids = [hit["id"] for hit in built["retrieved"]]
        self.assertIn("diabetes.this_eye", ids)
        self.assertIn("diabetes.other_eye", ids)
        self.assertIn("left eye", prompt)
        self.assertIn("Moderate=4", prompt)
        self.assertIn("Proliferative Diabetic Retinopathy (PDR)=6", prompt)
        self.assertIn("Other eye bilateral result", prompt)

    def test_other_eye_passage_is_absent_without_a_fellow_eye(self):
        built = build_screen_prompt("diabetes", DIABETES_PROMPT, existing_ads=2)
        ids = [hit["id"] for hit in built["retrieved"]]
        self.assertIn("diabetes.this_eye", ids)
        self.assertNotIn("diabetes.other_eye", ids)

    def test_unknown_condition_is_rejected(self):
        with self.assertRaises(ValueError):
            build_screen_prompt("myopia", DIABETES_PROMPT)


class GlaucomaPromptTests(unittest.TestCase):
    def test_measured_cdr_is_quoted_in_the_prompt(self):
        built = build_glaucoma_prompt(
            GLAUCOMA_PROMPT,
            geometric_cdr=0.62,
            existing_ads=16,
            other_eye_ads=17,
        )
        prompt = built["prompt"]
        self.assertTrue(prompt.startswith("[INST] <image>"))
        self.assertIn("0.620", prompt)
        self.assertIn("No Suspect or Negative=16", prompt)
        self.assertIn("Suspect or Positive=17", prompt)
        self.assertIn("Do not invent a different decimal", prompt)
        ids = [hit["id"] for hit in built["retrieved"]]
        self.assertIn("glaucoma.measurement", ids)
        self.assertIn("glaucoma.other_eye", ids)

    def test_missing_measurement_forbids_a_guess(self):
        built = build_glaucoma_prompt(GLAUCOMA_PROMPT, geometric_cdr=None)
        self.assertIn("unavailable", built["prompt"])
        self.assertIn("Do not invent a precise decimal", built["prompt"])


if __name__ == "__main__":
    unittest.main()
