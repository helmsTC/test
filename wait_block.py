async def get_queries_loop(name: str) -> bool:
    logger = _get_logger()
    aclient = await Client("localhost:6461", asynchronous=True)
    conn   = await utils.get_redis_connection()

    _start = time()

    # --- Prime the pump: fetch the very first pair ---
    state          = await _get_state(name)
    request_future = aclient.submit(get_queries, state, pure=False)  # loop=False by default

    for k in itertools.count():
        # 1) Wait for the next queries + scores
        try:
            queries, scores, stats = await request_future
        except CancelledError as e:
            logger.warning("get_queries cancelled for %s: %s", name, e)
            await asyncio.sleep(0.5)
            # try again from fresh state
            state          = await _get_state(name)
            request_future = aclient.submit(get_queries, state, pure=False)
            continue

        # 2) Sanity-check: we must have exactly two routes
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
            logger.warning("[%s] malformed routes payload, retrying: %r", name, current_routes)
            state          = await _get_state(name)
            request_future = aclient.submit(get_queries, state, pure=False)
            continue

        # 3) Auto-evaluation dispatch
        evalr = AUTO_EVAL.get(name, lambda _: None)
        if hasattr(evalr, "evaluate"):
            winner = evalr.evaluate(current_routes)
        elif callable(evalr):
            winner = evalr(current_routes)
        else:
            winner = None

        logger.info("[%s] Evaluator chose: %r", name, winner)

        if winner is not None:
            # --- Auto-update path ---
            answer = [{
                "ident":  current_routes[winner]["ident"],
                "pair":   [r["ident"] for r in current_routes],
                "winner": winner,
                "rank":   [2 if i == winner else 1 for i in range(2)],
                "routes": current_routes,
            }]
            logger.info("[%s] AUTO-updating model with answer %r", name, answer)
            try:
                # re-fetch state so we pick up any changes from extend_loop in flight
                state = await _get_state(name)
                fut   = aclient.submit(
                    update_model,
                    state,
                    answer,
                    pure=False,
                    key=f"{name}-auto-{k}"
                )
                await fut
            except (KeyError, CancelledError) as e:
                logger.warning("[%s] auto-update failed at iter %d: %s", name, k, e)

        else:
            # --- UI path: post to front-end and let extend_loop handle the user's click ---
            logger.info("[%s] LOW confidence → posting to UI for human feedback", name)
            await utils.post(name, (queries, scores, stats), delete=False)

            # **IMPORTANT**: wait for extend_loop to fire and update FUTURES[name].
            # A simple back-off here gives the other loop a chance to `extend`.
            await asyncio.sleep(0.5)

        # 4) Periodic bookkeeping & early exit
        if k % 10 == 0:
            uptime = time() - _start
            await conn.set(f"sampler-{name}-uptime", uptime)
        if await utils.should_stop(name):
            await asyncio.sleep(1)
            break

        # 5) Now fetch the *next* pair from the (new) state
        try:
            state = await _get_state(name)
        except KeyError as e:
            logger.exception("[%s] failed to reload state: %s", name, e)
            await asyncio.sleep(0.5)
        request_future = aclient.submit(get_queries, state, pure=False)

    # clean shutdown
    logger.info("[%s] stopping get_queries_loop; marking stopped flag", name)
    await conn.set(f"stopped-{name}-queries", b"1")
    return True
