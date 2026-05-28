import random

from locust import HttpUser, between, task


TEXT_PROMPTS = [
    "Create a short beginner home workout plan.",
    "Suggest a simple training session for a beginner.",
    "Prepare a safe full body workout.",
    "Give a short fitness recommendation.",
    "Create a simple plan for home training.",
]


class CoachApiUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        response = self.client.get("/info", name="/info")
        response.raise_for_status()
        self.info = response.json()
        self.input_type = self.info.get("input_type", "text")

    @task(1)
    def get_info(self):
        with self.client.get("/info", catch_response=True, name="/info") as response:
            body_preview = response.text[:500] if response.text else ""

            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}. Body: {body_preview}")
                return

            try:
                data = response.json()
            except Exception:
                response.failure(f"Response is not valid JSON. Body: {body_preview}")
                return

            if data.get("input_type") != "text":
                response.failure(f"Unexpected input_type: {data.get('input_type')}")
                return

            response.success()

    @task(10)
    def run_request(self):
        payload = {
            "content": random.choice(TEXT_PROMPTS),
            "extra_body": {},
        }

        with self.client.post("/run", json=payload, catch_response=True, name="/run") as response:
            body_preview = response.text[:500] if response.text else ""

            if response.status_code >= 400:
                response.failure(f"HTTP {response.status_code}. Body: {body_preview}")
                return

            try:
                data = response.json()
            except Exception:
                response.failure(
                    f"Response is not valid JSON. "
                    f"Content-Type: {response.headers.get('content-type')}. "
                    f"Body: {body_preview}"
                )
                return

            if data.get("status") != "success":
                response.failure(
                    f"Application error. Status: {data.get('status')}. "
                    f"Error: {data.get('error')}. "
                    f"Body: {body_preview}"
                )
                return

            response.success()