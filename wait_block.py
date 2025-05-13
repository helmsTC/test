async def get_queries_loop(name: str) -> bool:
    logger = _get_logger()
    aclient = await Client("localhost:6461", asynchronous=True)

    _start = time()

    # --- Prime the pump with the first pair ---
    state = await _get_state(name)
    request_future = aclient.submit(get_queries, state, loop=False, pure=False)

    for k in itertools.count():
        # 1) Await next batch of queries
        try:
            queries, scores, stats = await request_future
        except CancelledError as e:
            logger.warning("get_queries cancelled for %s: %s", name, e)
            await asyncio.sleep(0.5)
            # re-submit so we stay alive
            state = await _get_state(name)
            request_future = aclient.submit(get_queries, state, loop=False, pure=False)
            continue

        # 2) Validate we have exactly two routes
        current_routes = (
            queries[0].get("routes")
            if queries and isinstance(queries[0], dict)
            else None
        )
        valid_pair = (
            current_routes
            and len(current_routes) == 2
            and all(isinstance(r, dict) and "ident" in r for r in current_routes)
        )
        if not valid_pair:
            logger.info(f"[{name}] malformed routes, retrying: {current_routes!r}")
            state = await _get_state(name)
            request_future = aclient.submit(get_queries, state, loop=False, pure=False)
            continue

        # 3) Run the auto-evaluator (method vs. callable)
        evalr = AUTO_EVAL.get(name, lambda _: None)
        if hasattr(evalr, "evaluate"):
            winner = evalr.evaluate(current_routes)
        elif callable(evalr):
            winner = evalr(current_routes)
        else:
            winner = None

        logger.info(f"[{name}] Evaluator returned winner={winner}")

        if winner is not None:
            # --- Auto-update path (high confidence) ---
            answer = [{
                "ident":  current_routes[winner]["ident"],
                "pair":   [current_routes[0]["ident"], current_routes[1]["ident"]],
                "winner": winner,
                "rank":   [2 if i == winner else 1 for i in range(2)],
                "routes": current_routes,
            }]
            logger.info(f"[{name}] AUTO-updating model with answer={answer}")
            try:
                state = await _get_state(name)
                fut = aclient.submit(
                    update_model,
                    state,
                    answer,
                    pure=False,
                    key=f"{name}-auto-{k}"
                )
                await fut
            except (KeyError, CancelledError) as e:
                logger.warning("Auto-update failed for %s at iter %d: %s", name, k, e)

        else:
            # --- UI path (low confidence) ---
            logger.info(f"[{name}] Low confidence—posting to UI for human feedback")
            await utils.post(name, (queries, scores, stats), delete=False)

        # 4) Uptime bookkeeping & stop signal
        if k % 10 == 0:
            uptime = time() - _start
            await conn.set(f"sampler-{name}-uptime", uptime)
        if await utils.should_stop(name):
            await asyncio.sleep(1)
            break

        # 5) Re-submit to fetch the *next* pair (blocking until ready)
        try:
            state = await _get_state(name)
        except KeyError as e:
            logger.exception("Failed to reload state for %s: %s", name, e)
            await asyncio.sleep(0.5)
        request_future = aclient.submit(get_queries, state, loop=True, pure=False)

    # Clean shutdown
    logger.info("Stopping %s; setting stopped flag", name)
    await conn.set(f"stopped-{name}-queries", b"1")
    return True
