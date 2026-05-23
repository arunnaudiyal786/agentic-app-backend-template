---
name: ask
description: ONLY trigger this skill when the user explicitly types the slash command `/ask`. Do NOT trigger automatically for any other phrasing, context, or intent. When `/ask` is used, take the user's input and ask clarifying questions using the AskQuestion tool. Do NOT use any other skill — just ask questions.
---
 
# Ask Skill
 
## Trigger
This skill activates **only** when the user types `/ask` followed by their topic or question. Do NOT auto-trigger for any other phrasing or context.
 
 
A simple skill that takes the user's input and uses the `ask_user_input_v0` tool to ask clarifying questions for more context.
 
## Instructions
 
1. Read the user's input carefully.
2. Identify 1–3 things that would help you understand their intent, context, or constraints better.
3. Use the `ask_user_input_v0` tool to present those as clear, concise multiple-choice or short questions.
4. Do NOT invoke any other skill. Do NOT produce a final answer yet — just ask.
## Rules
 
- Ask only what's genuinely needed — don't over-ask.
- Keep questions short and options mutually exclusive.
- One question at a time if the topic is simple; up to 3 if the topic is complex.
- After the user answers, use their responses to proceed helpfully (or ask a follow-up if needed).
 
