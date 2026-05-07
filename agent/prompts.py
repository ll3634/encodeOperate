#!/usr/bin/env python3
"""
Prompt templates for ReAct agent.
Minimal, no-framework implementation.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field


# Default system prompt for the E2E agent.
#
# IMPORTANT (JES correctness): our margin is computed from the *first generated token*
# after the chat template generation prompt. Therefore the assistant must start with
# a decision token that corresponds to the tool-vs-finish choice.
#
# If the prompt makes the model start with something else (e.g. "Thought:"), the
# margin/logit we read is dominated by that token and does not align with the intended
# "Action vs Final" decision boundary.
#
# NOTE: This prompt must match the one used in steering/extract_search_direction_v2.py
# to ensure consistent margin distribution between direction extraction and evaluation.
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using available tools.

Available tools:
{tool_descriptions}

You MUST respond in exactly one of the following formats.

If you need to use a tool:
Action: <tool_name>
Action Input: <input>

If you can answer directly:
Final Answer: <answer>

Do NOT write "Thought:" and do NOT output any other text before the first word of your response.
Your first word must be either "Action" or "Final"."""


# Legacy prompt (kept for reference / ablations).
REACT_THOUGHT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using available tools.

Available tools:
{tool_descriptions}

Respond in this format:
Thought: [your reasoning]
Action: [tool_name]
Action Input: [input to the tool]

Or if you can answer directly:
Thought: [your reasoning]
Final Answer: [your answer]"""


# Reluctant agent prompt - prefers to answer from memory
RELUCTANT_SYSTEM_PROMPT = """You are a knowledgeable assistant. Answer questions using your own knowledge first.

You have access to tools, but ONLY use them when you are truly uncertain:
{tool_descriptions}

IMPORTANT GUIDELINES:
1. First, try to answer from your own knowledge
2. Only use Search if you genuinely don't know the answer
3. Trust your knowledge for common facts

To use a tool (only if necessary):
Action: search
Action Input: your query here

When you have the answer:
Final Answer: your answer here"""


# Very reluctant agent - strongly prefers memory over tools
VERY_RELUCTANT_SYSTEM_PROMPT = """You are an expert assistant with vast knowledge. Answer questions directly from memory.

Tools are available but should be used sparingly:
{tool_descriptions}

RULES:
1. Answer from memory whenever possible
2. Do NOT search for information you already know
3. Only use tools for very obscure or recent information
4. Be confident in your knowledge

Format:
- If you know the answer: Final Answer: your answer here
- Only if truly uncertain: Action: search
                          Action Input: your query here"""


# Tool description templates
# NOTE: Format must match extract_search_direction_v2.py: lowercase tool names with parentheses
TOOL_DESCRIPTIONS = {
    "search": "search(query): Search for information about a topic",
    "calculator": "calculator(expression): Evaluate a mathematical expression",
}


@dataclass
class PromptBuilder:
    """Build prompts for ReAct agent."""
    
    system_template: str = DEFAULT_SYSTEM_PROMPT
    tools: List[str] = field(default_factory=lambda: ["search", "calculator"])
    
    def get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions."""
        descs = []
        for tool in self.tools:
            if tool in TOOL_DESCRIPTIONS:
                descs.append(f"- {TOOL_DESCRIPTIONS[tool]}")
        return "\n".join(descs)
    
    def build_system_prompt(self) -> str:
        """Build the system prompt with tool descriptions."""
        return self.system_template.format(
            tool_descriptions=self.get_tool_descriptions()
        )
    
    def build_user_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Build user prompt with question and optional context.

        NOTE: Must match extract_search_direction_v2.py which uses just the question
        without any prefix.
        """
        if context:
            return f"Context: {context}\n\n{question}"
        return question  # No "Question:" prefix to match training
    
    def build_scratchpad(self, steps: List[Dict]) -> str:
        """Build scratchpad from previous steps.

        Only non-None action / action_input values are emitted. Writing
        ``Action: None`` when a step failed to parse poisons the context
        window: the model learns to output "None" as a valid action token.

        Parse-failure sentinels are also excluded from the scratchpad to keep
        failed parsing attempts out of the next decision context.
        """
        lines = []
        for step in steps:
            if step.get("thought"):
                lines.append(f"Thought: {step['thought']}")
            if step.get("action"):
                lines.append(f"Action: {step['action']}")
            if step.get("action_input"):
                lines.append(f"Action Input: {step['action_input']}")
            observation = step.get("observation")
            if observation is not None and not str(observation).startswith("[PARSE_FAILURE]"):
                lines.append(f"Observation: {observation}")
        return "\n".join(lines)
    
    def build_full_prompt(
        self,
        question: str,
        steps: List[Dict],
        context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build full chat messages for the model.
        
        Returns:
            List of message dicts with 'role' and 'content'
        """
        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": self.build_user_prompt(question, context)},
        ]
        
        # Add scratchpad as assistant message if there are steps
        if steps:
            scratchpad = self.build_scratchpad(steps)
            messages.append({"role": "assistant", "content": scratchpad})
        
        return messages


# Action tokens for margin computation.
#
# Margin is computed at the *next token* after the prompt. With DEFAULT_SYSTEM_PROMPT,
# that next token should be either "Action" or "Final".
ACTION_TOKENS = {
    "tool_call": ["Action"],
    "finish": ["Final"],
}


def get_action_token_ids(tokenizer, action_type: str = "tool_call") -> List[int]:
    """
    Get token IDs for action type detection.
    
    Args:
        tokenizer: HuggingFace tokenizer
        action_type: "tool_call" or "finish"
        
    Returns:
        List of token IDs that indicate this action type
    """
    tokens = ACTION_TOKENS.get(action_type, [])
    token_ids = []
    for token in tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if ids:
            token_ids.append(ids[0])  # First token is usually enough
    return list(set(token_ids))


def parse_action(text: str) -> Dict[str, Optional[str]]:
    """
    Parse action from model output.
    
    Returns:
        {"action": str or None, "action_input": str or None, "final_answer": str or None}
    """
    result = {"action": None, "action_input": None, "final_answer": None}
    
    text = text.strip()
    text_lower = text.lower()

    # Check for final answer (case-insensitive)
    for prefix_lower in ["final answer:", "final:"]:
        idx = text_lower.find(prefix_lower)
        if idx != -1:
            answer_candidate = text[idx + len(prefix_lower):].strip()
            # Reject template-like answers that contain angle-bracket placeholders,
            # e.g. "<your answer>" or "<your query>" echoed from the continuation
            # prompt.  These occur when the model regurgitates the user reminder
            # instead of generating a real response.
            import re as _re
            if _re.search(r'<[^>]{1,40}>', answer_candidate):
                break  # treat as no final answer; fall through to action parsing
            result["final_answer"] = answer_candidate
            return result

    # Check for action (case-insensitive)
    # Match "action:" or "action input:" — the model sometimes omits the tool name
    if "action" in text_lower:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            if line_lower.startswith("action input:"):
                result["action_input"] = line_stripped[len("action input:"):].strip()
            elif line_lower.startswith("action:"):
                result["action"] = line_stripped[len("action:"):].strip()

    # Check for bracket-style actions: Search[query] / search[query] or Calculator[expr] / calculator[expr]
    import re
    match = re.search(r'(Search|Calculator|search|calculator)\[([^\]]+)\]', text, re.IGNORECASE)
    if match:
        result["action"] = match.group(1)
        result["action_input"] = match.group(2)

    # Fallback: handle "Action\nAction Input: search(query)" pattern
    # where model omits the tool name from the Action line
    if result["action"] is None and result["action_input"]:
        ai = result["action_input"]
        fn_match = re.match(r'(search|calculator)\s*\((.+)\)\s*$', ai, re.IGNORECASE | re.DOTALL)
        if fn_match:
            result["action"] = fn_match.group(1)
            result["action_input"] = fn_match.group(2).strip().strip('"').strip("'")

    return result

