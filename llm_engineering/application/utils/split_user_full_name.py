from llm_engineering.domain.exceptions import ImproperlyConfigured


def split_user_full_name(user: str | None) -> tuple[str, str]:
    # split() with no argument drops extra spaces, so "" and "   " give an empty list.
    name_tokens = user.split() if user else []
    if len(name_tokens) == 0:
        raise ImproperlyConfigured("User name is empty")
    elif len(name_tokens) == 1:
        first_name, last_name = name_tokens[0], name_tokens[0]
    else:
        first_name, last_name = " ".join(name_tokens[:-1]), name_tokens[-1]

    return first_name, last_name
