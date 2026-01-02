from neural.systems.memory_store import MemoryStore


def test_memory_store_add_search(tmp_path):
    path = tmp_path / "memory.json"
    store = MemoryStore(str(path), max_entries=10)
    store.add("hello world", tags=["greeting"])
    store.add("brain simulation work", tags=["project"])

    results = store.search("hello")
    assert results
    assert results[0]["text"] == "hello world"


def test_memory_store_persistence(tmp_path):
    path = tmp_path / "memory.json"
    store = MemoryStore(str(path), max_entries=10)
    store.add("persist me", tags=["note"])
    store.save()

    loaded = MemoryStore(str(path), max_entries=10)
    assert loaded.entries
    assert loaded.entries[-1]["text"] == "persist me"
