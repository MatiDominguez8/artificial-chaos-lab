import json
from unittest.mock import patch

from src.acl.cognition import OllamaMemoryInterpreter
from src.acl.models import Agent, AgentProfile, Event
from src.acl.ollama_client import OllamaClient
from src.acl.policies import OllamaPolicy


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_ollama_policy_returns_structured_action_without_leaking_private_state():
    fake = FakeClient(
        {
            "action": "talk",
            "target": "Bruno",
            "amount": 0,
            "message": "Hola.",
            "reason": "Quiero conocerlo mejor.",
        }
    )
    policy = OllamaPolicy(client=fake, model="qwen3:14b")

    ada = Agent(
        "Ada",
        money=17,
        hunger=70,
        energy=42,
        working_memory=["Turn 4: Ada talked with Bruno."],
        profile=AgentProfile(
            traits={"risk_tolerance": 0.17, "curiosity": 0.83},
            goals=["Learn what Bruno wants."],
        ),
    )
    bruno = Agent(
        "Bruno",
        money=999,
        hunger=3,
        energy=91,
        profile=AgentProfile(
            traits={"risk_tolerance": 0.99},
            goals=["Keep a secret objective private."],
        ),
    )

    action = policy.choose_action(ada, [bruno])

    assert action.actor == "Ada"
    assert action.kind.value == "talk"
    assert action.target == "Bruno"
    assert action.reason == "Quiero conocerlo mejor."

    prompt = fake.calls[0]["messages"][1]["content"]
    assert "money: 17" in prompt
    assert "hunger: 70/100" in prompt
    assert "Bruno: relationship=+0.00" in prompt
    assert "risk_tolerance: 0.17" in prompt
    assert "curiosity: 0.83" in prompt
    assert "Learn what Bruno wants." in prompt
    assert "Turn 4: Ada talked with Bruno." in prompt
    assert "999" not in prompt
    assert "hunger: 3" not in prompt
    assert "risk_tolerance: 0.99" not in prompt
    assert "Keep a secret objective private." not in prompt


def test_ollama_client_sends_nothink_and_json_schema():
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "action": "rest",
                                "target": "",
                                "amount": 0,
                                "message": "",
                                "reason": "Estoy cansado.",
                            }
                        )
                    }
                }
            ).encode("utf-8")

    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    schema = {
        "type": "object",
        "properties": {"action": {"type": "string"}},
        "required": ["action"],
    }

    with patch("src.acl.ollama_client.urlopen", fake_urlopen):
        result = OllamaClient(timeout=12).chat(
            model="qwen3:14b",
            messages=[{"role": "user", "content": "test"}],
            schema=schema,
            temperature=0.8,
            num_ctx=4096,
        )

    assert captured["payload"]["think"] is False
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["format"] == schema
    assert captured["payload"]["options"]["num_ctx"] == 4096
    assert result["action"] == "rest"



def test_ollama_memory_interpreter_can_change_relationship_from_message():
    fake = FakeClient(
        {
            "remember": True,
            "content": "Bruno offered to cooperate with me.",
            "importance": 0.7,
            "emotional_intensity": 0.4,
            "confidence": 0.9,
            "relationship_delta": 0.12,
            "belief": "Bruno may be cooperative.",
            "belief_confidence": 0.75,
        }
    )
    observer = Agent(
        "Ada",
        profile=AgentProfile(
            traits={"trustfulness": 0.4},
            goals=["Stay secure."],
        ),
    )
    event = Event(
        turn=1,
        actor="Bruno",
        kind="talk",
        summary='Bruno said to Ada: "Let us cooperate."',
        target="Ada",
        message="Let us cooperate.",
    )

    decision = OllamaMemoryInterpreter(
        client=fake,
        model="qwen3:14b",
    ).interpret(observer, event)

    assert decision.remember is True
    assert decision.subject == "Bruno"
    assert decision.relationship_delta == 0.12
    assert decision.belief == "Bruno may be cooperative."
    prompt = fake.calls[0]["messages"][1]["content"]
    assert "Let us cooperate." in prompt
