import os


os.environ["LOCAL_AUTH_USERNAME"] = "tests@example.com"
os.environ["LOCAL_AUTH_PASSWORD"] = "test-only-password-123"
os.environ["LOCAL_AUTH_SECRET"] = "test-only-session-signing-secret-1234567890"
