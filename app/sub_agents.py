# app/sub_agents.py

class BaseAgent:
    """Base class for all therapeutic sub-agents."""
    
    def handle(self, user_input, emotion):
        """Process user input and emotion to generate a response.
        
        Args:
            user_input: The text input from the user
            emotion: The detected emotion from the classifier
            
        Returns:
            A string containing the agent's response
        """
        raise NotImplementedError("Sub-agents must implement the handle method")


class CBTAgent(BaseAgent):
    """Cognitive Behavioral Therapy agent.
    
    Focuses on identifying and challenging negative thought patterns.
    """
    
    def handle(self, user_input, emotion):
        return f"[CBT] Let's try to reframe that thought. Can you identify any distortions in: '{user_input}'?"


class PsychoanalysisAgent(BaseAgent):
    """Psychoanalysis therapy agent.
    
    Focuses on exploring past experiences and unconscious patterns.
    """
    
    def handle(self, user_input, emotion):
        return f"[Psychoanalysis] That sounds like it may be rooted in past experiences. Tell me more about why you think you feel '{emotion}' in this moment."


class MotivationalAgent(BaseAgent):
    """Motivational therapy agent.
    
    Focuses on strengths and positive reinforcement.
    """
    
    def handle(self, user_input, emotion):
        return f"[Motivational] You're doing better than you think. Let's look at the progress you've made despite feeling '{emotion}'."


class CrisisAgent(BaseAgent):
    """Crisis management agent.
    
    Focuses on immediate stabilization and safety.
    """
    
    def handle(self, user_input, emotion):
        return f"[Crisis] I'm here with you. It sounds intense. You are not alone—let's focus on taking one step at a time."


# Agent registry - maps agent types to their implementations
sub_agents = {
    "CBT": CBTAgent(),
    "Psychoanalysis": PsychoanalysisAgent(),
    "Motivational": MotivationalAgent(),
    "Crisis": CrisisAgent(),
}


# Routing logic based on detected emotions
def route_to_agent(emotion):
    """Route to the appropriate agent based on detected emotion.
    
    Args:
        emotion: The primary emotion detected in the user's message
        
    Returns:
        The key of the selected agent from the sub_agents dictionary
    """
    # Emotions that benefit from cognitive restructuring
    if emotion in ["sadness", "disappointment", "grief"]:
        return "CBT"
    
    # Emotions that benefit from deeper exploration
    elif emotion in ["anger", "annoyance", "frustration", "disgust"]:
        return "Psychoanalysis"
    
    # Positive emotions that can be reinforced
    elif emotion in ["joy", "caring", "excitement", "optimism"]:
        return "Motivational"
    
    # Emotions indicating immediate support needs
    elif emotion in ["fear", "nervousness", "anxiety", "stress"]:
        return "Crisis"
    
    # Default to CBT for other emotions or neutral
    else:
        return "CBT"  # safe fallback