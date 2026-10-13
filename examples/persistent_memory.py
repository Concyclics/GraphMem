"""Run twice against the same directory: the message count stays at one."""
import argparse

from graphmem import MemoryService, ServiceConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./data/example-memory")
    args = parser.parse_args()
    with MemoryService(ServiceConfig.offline(args.data_dir)) as memory:
        print(memory.add("alice", [{"role": "user", "speaker": "Alice",
                                    "content": "I adopted a dog named Luna."}],
                         tenant_id="demo", session_id="chat-001", request_id="event-001"))
        request = memory.prepare("alice", "What is the name of Alice's dog?", tenant_id="demo")
        print("Evidence:", request.evidence_turn_ids)
        print("Prompt SHA256:", request.prompt_payload_hash)
    with MemoryService(ServiceConfig.offline(args.data_dir)) as reopened:
        assert reopened.status("alice", tenant_id="demo")["turn_count"] == 1
        print("Persistence verified after reopening the database.")


if __name__ == "__main__":
    main()
