"""Architect for managing TinyROS nodes."""


class Architect:
    """Context manager to wire up and manage the lifecycle of TinyROS nodes.

    Usage:
        with Architect(node1, node2, node3):
            # Within this block, all nodes are connected (publishers -> subscribers)
            # and running. When the block exits, all nodes are asked to stop.
    """

    def __init__(self, *nodes):
        """Initialize the Architect with a variable number of nodes.

        Args:
            *nodes: A variable number of Node instances to be managed by the Architect.
        """
        self._nodes = nodes

    def __enter__(self):
        """Connect every publisher to every subscriber that requested its topic."""
        for publisher in self._nodes:
            for topic in getattr(publisher, "_publishers", {}):
                # For each other node, see if it subscribed to this topic
                for subscriber in self._nodes:
                    if subscriber is publisher:
                        continue
                    if topic in getattr(subscriber, "_requested_subscriptions", []):
                        buffer = publisher.get_connection_to_publisher(topic)
                        subscriber.connect_subscriber(topic, buffer)

        # Start all nodes
        for node in self._nodes:
            node.start()

        # Return self in case the user wants to inspect or access nodes
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Request all nodes to stop and clean up resources.

        Args:
            exc_type: The type of exception raised, if any.
            exc_val: The value of the exception raised, if any.
            exc_tb: The traceback object, if an exception was raised.
        """
        # Ask each node to stop
        for node in self._nodes:
            try:
                node.request_stop()
            except Exception:
                pass

        # If a node defines a join() method, wait for it to terminate
        for node in self._nodes:
            join_fn = getattr(node, "join", None)
            if callable(join_fn):
                try:
                    join_fn()
                except Exception:
                    pass
