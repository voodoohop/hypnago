"""
Flatten a ComfyUI UI workflow (with subgraphs/components) into API prompt format.

ComfyUI's API expects a flat dict of {node_id: {class_type, inputs}} but the UI
format can contain embedded subgraph components. This script expands them.
"""

import json
import sys


def flatten_workflow(wf):
    """Convert UI workflow format to API prompt format, expanding subgraphs."""
    # Build subgraph lookup
    subgraphs = {}
    for sg in wf.get("definitions", {}).get("subgraphs", []):
        subgraphs[sg["id"]] = sg

    # Build link lookup from the main workflow: link_id -> [src_node, src_slot, dst_node, dst_slot, type]
    link_map = {}
    for link in wf.get("links", []):
        link_id = link[0]
        link_map[link_id] = link

    prompt = {}
    # Track node ID offsets for subgraph nodes to avoid collisions
    next_id = wf.get("last_node_id", 1000) + 1

    def process_node(node, parent_links=None, input_mapping=None):
        nonlocal next_id
        node_type = node["type"]
        node_id = node["id"]

        # Check if this node is a subgraph reference
        if node_type in subgraphs:
            sg = subgraphs[node_type]
            expand_subgraph(node, sg, parent_links or link_map, input_mapping)
            return

        # Regular node — convert to API format
        api_node = {"class_type": node_type, "inputs": {}}

        # Process widget values
        widgets = node.get("widgets_values", {})
        if isinstance(widgets, dict):
            for k, v in widgets.items():
                if k != "videopreview":
                    api_node["inputs"][k] = v
        elif isinstance(widgets, list):
            # Widget values are positional — we need the node info to map them
            # For now, just pass the node; ComfyUI will use defaults
            pass

        # Process linked inputs
        for inp in node.get("inputs", []):
            name = inp["name"]
            link_id = inp.get("link")
            if link_id is not None:
                links = parent_links or link_map
                if link_id in links:
                    link = links[link_id]
                    src_node_id = link[1]
                    src_slot = link[2]
                    # Check input_mapping for subgraph inputs
                    if input_mapping and src_node_id in input_mapping:
                        api_node["inputs"][name] = input_mapping[src_node_id]
                    else:
                        api_node["inputs"][name] = [str(src_node_id), src_slot]
            elif "widget" in inp:
                # Widget-connected input — value comes from widgets_values
                pass

        prompt[str(node_id)] = api_node

    def expand_subgraph(parent_node, sg, parent_link_map, parent_input_mapping):
        nonlocal next_id

        # Map subgraph input IDs to the actual source connections from the parent
        input_mapping = {}
        sg_inputs = sg.get("inputs", [])
        parent_inputs = parent_node.get("inputs", [])

        # Build inner link map
        inner_link_map = {}
        for link in sg.get("links", []):
            link_id = link[0]
            inner_link_map[link_id] = link

        # Map subgraph input connections: for each parent input, find what it connects to inside
        for pi in parent_inputs:
            pi_name = pi["name"]
            pi_link = pi.get("link")
            if pi_link is None:
                continue

            # Find the corresponding subgraph input definition
            sg_input = None
            for si in sg_inputs:
                if si["name"] == pi_name:
                    sg_input = si
                    break

            if sg_input and pi_link in parent_link_map:
                parent_link = parent_link_map[pi_link]
                src_node = parent_link[1]
                src_slot = parent_link[2]
                # The subgraph input feeds into inner links
                for inner_link_id in sg_input.get("linkIds", []):
                    if inner_link_id in inner_link_map:
                        inner_link = inner_link_map[inner_link_id]
                        # Override the source of this inner link
                        inner_link[1] = src_node
                        inner_link[2] = src_slot

        # Map subgraph output connections
        sg_outputs = sg.get("outputs", [])
        parent_outputs = parent_node.get("outputs", [])

        # Process inner nodes
        for inner_node in sg.get("nodes", []):
            process_node(inner_node, inner_link_map, input_mapping)

        # Wire output: find what the parent's output links connect to, and update
        for idx, so in enumerate(sg_outputs):
            for link_id in so.get("linkIds", []):
                if link_id in inner_link_map:
                    inner_link = inner_link_map[link_id]
                    output_src_node = inner_link[1]
                    output_src_slot = inner_link[2]
                    # Update any parent links that read from this subgraph's output
                    if idx < len(parent_outputs):
                        po = parent_outputs[idx]
                        for parent_out_link_id in (po.get("links") or []):
                            if parent_out_link_id in parent_link_map:
                                parent_link_map[parent_out_link_id][1] = output_src_node
                                parent_link_map[parent_out_link_id][2] = output_src_slot

    # Process all top-level nodes
    for node in wf.get("nodes", []):
        process_node(node, link_map)

    return prompt


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        wf = json.load(f)

    prompt = flatten_workflow(wf)
    print(json.dumps(prompt, indent=2))
