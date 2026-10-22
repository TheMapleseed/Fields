"""A small KDL reader for this project's configuration files."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    name: str
    args: tuple[object, ...]
    children: tuple["Node", ...]


class KdlError(ValueError):
    pass


def parse_kdl(text: str) -> tuple[Node, ...]:
    tokens = _tokenize(text)
    nodes, index = _parse_nodes(tokens, 0, stop_on_rbrace=False)
    if index != len(tokens):
        raise KdlError("unexpected trailing tokens")
    return nodes


def _parse_nodes(
    tokens: list[_Token],
    index: int,
    stop_on_rbrace: bool,
) -> tuple[tuple[Node, ...], int]:
    nodes: list[Node] = []
    while index < len(tokens):
        token = tokens[index]
        if token.kind == "newline":
            index += 1
            continue
        if stop_on_rbrace and token.kind == "rbrace":
            break
        if token.kind != "ident":
            raise KdlError(f"expected a node name, found {token.kind}")
        name = str(token.value)
        index += 1
        args: list[object] = []
        while index < len(tokens) and tokens[index].kind in {"string", "number"}:
            args.append(tokens[index].value)
            index += 1
        children: tuple[Node, ...] = ()
        if index < len(tokens) and tokens[index].kind == "lbrace":
            index += 1
            children, index = _parse_nodes(tokens, index, stop_on_rbrace=True)
            if index >= len(tokens) or tokens[index].kind != "rbrace":
                raise KdlError(f"node {name} is missing a closing brace")
            index += 1
        nodes.append(Node(name, tuple(args), children))
        if index < len(tokens) and tokens[index].kind == "newline":
            index += 1
    return tuple(nodes), index


@dataclass(frozen=True)
class _Token:
    kind: str
    value: object


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char in " \t\r":
            index += 1
            continue
        if char == "\n":
            tokens.append(_Token("newline", None))
            index += 1
            continue
        if char == "/" and index + 1 < length and text[index + 1] == "/":
            index += 2
            while index < length and text[index] != "\n":
                index += 1
            continue
        if char == "/" and index + 1 < length and text[index + 1] == "*":
            end = text.find("*/", index + 2)
            if end < 0:
                raise KdlError("unterminated block comment")
            index = end + 2
            continue
        if char == "{":
            tokens.append(_Token("lbrace", None))
            index += 1
            continue
        if char == "}":
            tokens.append(_Token("rbrace", None))
            index += 1
            continue
        if char == '"':
            value, index = _string(text, index)
            tokens.append(_Token("string", value))
            continue
        if char == "-" or char.isdigit():
            value, index = _number(text, index)
            tokens.append(_Token("number", value))
            continue
        if char.isalpha() or char == "_":
            start = index
            index += 1
            while index < length and (text[index].isalnum() or text[index] in "_-"):
                index += 1
            tokens.append(_Token("ident", text[start:index]))
            continue
        raise KdlError(f"unexpected character {char!r}")
    return tokens


def _string(text: str, index: int) -> tuple[str, int]:
    index += 1
    chars: list[str] = []
    escapes = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}
    while index < len(text):
        char = text[index]
        if char == '"':
            return "".join(chars), index + 1
        if char == "\\":
            if index + 1 >= len(text) or text[index + 1] not in escapes:
                raise KdlError("invalid string escape")
            chars.append(escapes[text[index + 1]])
            index += 2
            continue
        if char == "\n":
            raise KdlError("unterminated string")
        chars.append(char)
        index += 1
    raise KdlError("unterminated string")


def _number(text: str, index: int) -> tuple[int | float, int]:
    start = index
    if text[index] == "-":
        index += 1
    if index >= len(text) or not text[index].isdigit():
        raise KdlError("invalid number")
    while index < len(text) and text[index].isdigit():
        index += 1
    is_float = False
    if index < len(text) and text[index] == ".":
        is_float = True
        index += 1
        if index >= len(text) or not text[index].isdigit():
            raise KdlError("invalid number")
        while index < len(text) and text[index].isdigit():
            index += 1
    if index < len(text) and text[index] in "eE":
        is_float = True
        index += 1
        if index < len(text) and text[index] in "+-":
            index += 1
        if index >= len(text) or not text[index].isdigit():
            raise KdlError("invalid number")
        while index < len(text) and text[index].isdigit():
            index += 1
    raw = text[start:index]
    return (float(raw) if is_float else int(raw)), index
