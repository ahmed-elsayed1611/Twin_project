from threading import Lock
from typing import ClassVar


class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """

    _instances: ClassVar = {}
    _init_arguments: ClassVar = {}

    _lock: Lock = Lock()

    """
    We now have a lock object that will be used to synchronize threads during
    first access to the Singleton.
    """

    def __call__(cls, *args, **kwargs):
        """
        Returns the one instance of the class. Calling it again with different
        `__init__` arguments raises instead of silently returning the old instance.
        """
        # Now, imagine that the program has just been launched. Since there's no
        # Singleton instance yet, multiple threads can simultaneously pass the
        # previous conditional and reach this point almost at the same time. The
        # first of them will acquire lock and will proceed further, while the
        # rest will wait here.
        with cls._lock:
            # The first thread to acquire the lock, reaches this conditional,
            # goes inside and creates the Singleton instance. Once it leaves the
            # lock block, a thread that might have been waiting for the lock
            # release may then enter this section. But since the Singleton field
            # is already initialized, the thread won't create a new object.
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
                cls._init_arguments[cls] = (args, kwargs)
            elif (args or kwargs) and (args, kwargs) != cls._init_arguments[cls]:
                # Returning the old instance here would silently ignore the new arguments (e.g. another model_id).
                raise ValueError(
                    f"{cls.__name__} already exists with arguments {cls._init_arguments[cls]}; got {(args, kwargs)}."
                )

        return cls._instances[cls]
