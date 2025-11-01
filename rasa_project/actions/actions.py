# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

# rasa_project/actions/actions.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from transformers_module.transformer_brain import generate_empathic_reply
from context_engine.memory import add_to_memory, get_relevant_context
import logging

logger = logging.getLogger(__name__)

class ActionEmpathicReply(Action):
    def name(self) -> Text:
        return "action_empathic_reply"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_msg = tracker.latest_message.get('text')
        # Retrieve context from memory
        context = get_relevant_context(user_msg, top_k=3)
        # Generate an empathic reply using the transformer module
        reply = generate_empathic_reply(user_msg, context)

        # Store user message and bot reply in memory
        add_to_memory({"role":"user","text":user_msg})
        add_to_memory({"role":"assistant","text":reply})

        dispatcher.utter_message(text=reply)
        return []

class ActionProvideTip(Action):
    def name(self) -> Text:
        return "action_provide_tip"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # simple canned tips OR call transformer to craft a gentle step
        tips = "Try breathing in for 4 sec, hold 4 sec, out 4 sec. Would you like to try now?"
        dispatcher.utter_message(text=tips)
        add_to_memory({"role":"assistant","text":tips})
        return []

class ActionHandleCrisis(Action):
    def name(self) -> Text:
        return "action_handle_crisis"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict):
        # Mandatory safety behavior: clear, empathic, and encourage seeking help + emergency resources
        message = ("I’m really sorry that you’re feeling this way. I’m not able to help with emergencies.\n"
                   "Please contact your local emergency services or a crisis hotline right now. "
                   "If you’re in India, dial 112; if you're elsewhere, please contact your local emergency number.")
        dispatcher.utter_message(text=message)
        # optionally log or flag for human follow up
        return []
