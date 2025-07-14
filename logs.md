INFO:     127.0.0.1:51579 - "POST /api/profiling/clients/d8f627e8-5a6d-4e4c-911c-49f06256410c/analyze-needs HTTP/1.1" 500 Internal Server Error
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "/Users/ivanculo/Desktop/Projects/MyMind/.venv/lib/python3.9/site-packages/uvicorn/protocols/http/httptools_impl.py", line 426, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
  File "/Users/ivanculo/Desktop/Projects/MyMind/.venv/lib/python3.9/site-packages/uvicorn/middleware/proxy_headers.py", line 84, in __call__
    return await self.app(scope, receive, send)
  File "/Users/ivanculo/Desktop/Projects/MyMind/.venv/lib/python3.9/site-packages/fastapi/applications.py", line 1106, in __call__
    await super().__call__(scope, receive, send)
  File "/Users/ivanculo/Desktop/Projects/MyMind/.venv/lib/python3.9/site-packages/starlette/applications.py", line 122, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/Users/ivanculo/Desktop/Projects/MyMind/.venv/lib/python3.9/site-packages/starlette/middleware/errors.py", line 184, in __call__
    raise exc
  File "/Users/ivanculo/Desktop/Projects/MyMind/.venv/lib/python3.9/site-packages/starlette/middleware/errors.py", line 162, in __call__
    await self.app(scope, receive, _send)
  File "/Users/ivanculo/Desktop/Projects/MyMind/.venv/lib/python3.9/site-packages/starlette/middleware/exceptions.py", line 79, in __call__
    raise exc
  File "/Users/ivanculo/Desktop/Projects/MyMind/.venv/lib/python3.9/site-packages/starlette/middleware/exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "/Users/ivanculo/Desktop/Projects/MyMind/.venv/lib/python3.9/site-packages/fastapi/middleware/asyncexitstack.py", line 20, in __call__
    raise e
  File "/Users/ivanculo/Desktop/Projects/MyMind/.venv/lib/python3.9/site-packages/fastapi/middleware/asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "/Users/ivanculo/Desktop/Projects/MyMind/.venv/lib/python3.9/site-packages/starlette/routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "/Users/ivanculo/Desktop/Projects/MyMind/.venv/lib/python3.9/site-packages/starlette/routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "/Users/ivanculo/Desktop/Projects/MyMind/.venv/lib/python3.9/site-packages/starlette/routing.py", line 66, in app
    response = await func(request)
  File "/Users/ivanculo/Desktop/Projects/MyMind/.venv/lib/python3.9/site-packages/fastapi/routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "/Users/ivanculo/Desktop/Projects/MyMind/.venv/lib/python3.9/site-packages/fastapi/routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "/Users/ivanculo/Desktop/Projects/MyMind/src/api/routers/profiling.py", line 134, in analyze_client_needs
    session_ids = get_recent_sessions(client_id, session_count, db)
  File "/Users/ivanculo/Desktop/Projects/MyMind/src/api/routers/profiling.py", line 28, in get_recent_sessions
    return [session.id for session in sessions]
  File "/Users/ivanculo/Desktop/Projects/MyMind/src/api/routers/profiling.py", line 28, in <listcomp>
    return [session.id for session in sessions]
AttributeError: 'UUID' object has no attribute 'id'