"""Load testing scenarios for ML Pipeline Monitor API using Locust."""

from __future__ import annotations

from locust import HttpUser, between, task


class MLMonitorUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        resp = self.client.post(
            "/v1/auth/login",
            json={"username": "admin", "password": "password", "refresh": False},
        )
        if resp.status_code == 200:
            self.token = resp.json().get("access_token", "")
        else:
            self.token = ""

    @task(3)
    def health_check(self):
        self.client.get("/health")

    @task(5)
    def predict(self):
        if not self.token:
            return
        self.client.post(
            "/v1/predict",
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "features": {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2},
                "dataset": "Iris Species",
            },
        )

    @task(1)
    def metrics(self):
        self.client.get("/metrics")
