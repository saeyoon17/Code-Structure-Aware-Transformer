import string
from typing import List

identifier_type = {
    "java": ["identifier"],
    "python": ["identifier", "list_splat_pattern", "type_conversion"],
    "ruby": [
        "identifier",
        "hash_key_symbol",
        "simple_symbol",
        "constant",
        "instance_variable",
        "global_variable",
        "class_variable",
    ],
    "javascript": [
        "identifier",
        "hash_key_symbol",
        "simple_symbol",
        "constant",
        "instance_variable",
        "global_variable",
        "class_variable",
        "property_identifier",
        "shorthand_property_identifier",
        "statement_identifier",
        "shorthand_property_identifier_pattern",
        "regex_flags",
    ],
    "go": [
        "identifier",
        "hash_key_symbol",
        "simple_symbol",
        "constant",
        "instance_variable",
        "global_variable",
        "class_variable",
        "property_identifier",
        "shorthand_property_identifier",
        "statement_identifier",
        "shorthand_property_identifier_pattern",
        "regex_flags",
        "type_identifier",
        "field_identifier",
        "package_identifier",
        "label_name",
    ],
}
string_type = {
    "java": ["string", "comment", "string_literal", "character_literal"],
    "python": [
        "heredoc_content",
        "string",
        "comment",
        "string_literal",
        "character_literal",
        "chained_string",
        "escape_sequence",
    ],
    "ruby": [
        "heredoc_content",
        "string",
        "comment",
        "string_literal",
        "character_literal",
        "chained_string",
        "escape_sequence",
        "string_content",
        "heredoc_beginning",
        "heredoc_end",
    ],
    "javascript": [
        "heredoc_content",
        "string",
        "comment",
        "string_literal",
        "character_literal",
        "chained_string",
        "escape_sequence",
        "string_content",
        "heredoc_beginning",
        "heredoc_end",
        "jsx_text",
        "regex_pattern",
        "string_fragment",
    ],
    "go": [
        "heredoc_content",
        "string",
        "comment",
        "string_literal",
        "character_literal",
        "chained_string",
        "escape_sequence",
        "string_content",
        "heredoc_beginning",
        "heredoc_end",
        "regex_pattern",
        "\n",
        "raw_string_literal",
        "rune_literal",
    ],
}


def split_identifier_into_parts(identifier: str) -> List[str]:
    """
    Split a single identifier into parts on snake_case and camelCase
    come from code transformer
    """
    snake_case = identifier.split("_")

    identifier_parts = []  # type: List[str]
    for i in range(len(snake_case)):
        part = snake_case[i]
        if len(part) > 0:
            identifier_parts.extend(s.lower() for s in split_camelcase(part))
    if len(identifier_parts) == 0:
        return [identifier]
    return identifier_parts


def judge_func(source, target):
    """
    value=   <==>     value
    tree sitter will remove = after value, so cant find func name
    :param source:
    :param target:
    :return:
    """
    if source == target:
        return True
    s, t = split_word(source), split_word(target)
    if s == t:
        return True
    return False


def is_number(s: str):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata

        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def split_camelcase(camel_case_identifier: str) -> List[str]:
    """
    Split camelCase identifiers.
    come from code transformer
    """
    if not len(camel_case_identifier):
        return []
    # split into words based on adjacent cases being the same
    result = []
    current = str(camel_case_identifier[0])
    prev_upper = camel_case_identifier[0].isupper()
    prev_digit = camel_case_identifier[0].isdigit()
    prev_special = not camel_case_identifier[0].isalnum()
    for c in camel_case_identifier[1:]:
        upper = c.isupper()
        digit = c.isdigit()
        special = not c.isalnum()
        new_upper_word = upper and not prev_upper
        new_digit_word = digit and not prev_digit
        new_special_word = special and not prev_special
        if new_digit_word or new_upper_word or new_special_word:
            result.append(current)
            current = c
        elif not upper and prev_upper and len(current) > 1:
            result.append(current[:-1])
            current = current[-1] + c
        elif not digit and prev_digit:
            result.append(current)
            current = c
        elif not special and prev_special:
            result.append(current)
            current = c
        else:
            current += c
        prev_digit = digit
        prev_upper = upper
        prev_special = special
    result.append(current)
    return result


# Performs a dfs traversl on ast
def dfs_graph(code, data_lines, node, graph, idx, node_lst, num_node, language):
    node_name = None
    # why tptrans does not have to do this? --> it does not have to remove all nodes, as it only usese path. But we have to.It only removes them from terminals
    if node.type in string.punctuation:
        return node_name, idx, node_lst
    idx = idx + 1
    node_name = "nont:" + node.type + f":{node.start_point[0]}" + f":{node.end_point[0]}" + f":{idx}"
    graph.add_node(node_name)
    node_lst.append(node_name)

    # Do not add node if node is str type.
    if len(node.children) == 0:
        if node.type in string_type[language]:
            pass
        else:
            l_, r_ = node.start_point, node.end_point
            if l_[0] != r_[0]:
                print(f"start and end lines are different for a literal")
                assert False

            literal = data_lines[l_[0]][l_[1] : r_[1]]
            blocks = split_identifier_into_parts(literal)

            # Fix: if it is nonterminal leaf, add terminla node

            if node.type in identifier_type[language]:
                parent_name = node_name
                for idt_split in blocks:
                    idx = idx + 1
                    identifier_split_name = "idt:" + idt_split + f":{node.start_point[0]}" + f":{node.end_point[0]}" + f":{idx}"
                    graph.add_node(identifier_split_name)
                    node_lst.append(identifier_split_name)
                    graph.add_edge(parent_name, identifier_split_name)
                    parent_name = identifier_split_name
            # do not add num token.
            elif is_number(literal) or node.type in [
                "decimal_integer_literal",
                "decimal_floating_point_literal",
                "hex_integer_literal",
                "integer",
                "float",
                "int_literal",
                "imaginary_literal",
                "float_literal",
            ]:
                pass
            # # do not add punctuations
            elif literal in string.punctuation:
                pass
            # # otherwise, just add
            else:
                idx = idx + 1
                name = "idt:" + literal + f":{node.start_point[0]}" + f":{node.end_point[0]}" + f":{idx}"
                # print(name)
                graph.add_node(name)
                node_lst.append(name)
                graph.add_edge(node_name, name)
                num_node += 1
                pass

    for child_idx, child in enumerate(node.children):
        child_name, new_idx, new_node_lst = dfs_graph(
            code,
            data_lines,
            child,
            graph,
            idx,
            node_lst,
            num_node,
            language,
        )
        idx = new_idx
        node_lst = new_node_lst
        if child_name != None:
            graph.add_edge(node_name, child_name)
    return node_name, idx, node_lst
