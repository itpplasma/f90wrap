#!/usr/bin/env python3
import sys
import test  # The Direct-C extension

# Test calling foo
try:
    test.foo(3.14, 42)
    print("SUCCESS: Direct-C call completed")
    sys.exit(0)
except Exception as exc:  # pragma: no cover - manual smoke test
    print(f"FAIL: {exc}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
