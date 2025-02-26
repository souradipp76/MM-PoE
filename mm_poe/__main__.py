"""Entry point for mm_poe."""


def entry_point():
    from .cli import main

    main()


if __name__ == "__main__":
    entry_point()
