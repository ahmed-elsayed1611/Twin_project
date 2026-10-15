import opik


def _no_track(*args, **kwargs):
    """Unit tests never send traces to Opik. This Opik version has no setting to turn tracking off, and
    `@opik.track` runs when a module is imported, so it is replaced before any test module is collected."""

    if args and callable(args[0]):
        return args[0]

    return lambda function: function


opik.track = _no_track
