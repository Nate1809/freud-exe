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
    
    # Enhanced meta_intent for stronger personality
    meta_intent = "insightful and grounding"
    
    def handle(self, user_input, emotion):
        return f"I notice patterns in how you're framing this experience. The {emotion} you're feeling might be connected to underlying thought patterns. Let's explore what beliefs might be beneath the surface and see if we can find a perspective that feels both true and less painful."


class PsychoanalysisAgent(BaseAgent):
    """Psychoanalysis therapy agent.
    
    Focuses on exploring past experiences and unconscious patterns.
    """
    
    # Enhanced meta_intent for stronger personality
    meta_intent = "deeply reflective and gently probing"
    
    def handle(self, user_input, emotion):
        return f"The {emotion} you're experiencing now feels significant. There's often a deeper story connected to these feelings - perhaps echoes from past experiences or unconscious patterns. I'm curious about what might be beneath this feeling that we could explore together."


class MotivationalAgent(BaseAgent):
    """Motivational therapy agent.
    
    Focuses on strengths and positive reinforcement.
    """
    
    # Enhanced meta_intent for stronger personality
    meta_intent = "genuinely encouraging and empowering"
    
    def handle(self, user_input, emotion):
        return f"Even in moments of {emotion}, I can see strength in how you're engaging with these feelings. There's resilience in you that might be easy to overlook right now. Let's acknowledge what's working and build from there - you have more capacity than you might be giving yourself credit for."


class CrisisAgent(BaseAgent):
    """Crisis management agent.
    
    Focuses on immediate stabilization and safety.
    """
    
    # Enhanced meta_intent for stronger personality
    meta_intent = "steadfast, reassuring, and present"
    
    def handle(self, user_input, emotion):
        return f"I'm right here with you through this intense {emotion}. You don't have to face this alone. Let's focus first on what you need right now, in this moment, to help you feel even slightly more grounded. We'll take this one step at a time together."


# Default fallback agent
class DefaultAgent(BaseAgent):
    """Default fallback agent.
    
    Provides general support when no specific approach is indicated.
    """
    
    # Enhanced meta_intent for stronger personality
    meta_intent = "warm, present, and attentively supportive"
    
    def handle(self, user_input, emotion):
        return f"I'm here with you, and I'm listening deeply. There's a lot happening in what you've shared, and I want to understand more about your experience and how you're making sense of it."


# Agent registry - maps agent types to their implementations
sub_agents = {
    "CBT": CBTAgent(),
    "Psychoanalysis": PsychoanalysisAgent(),
    "Motivational": MotivationalAgent(),
    "Crisis": CrisisAgent(),
    # Default agent as fallback
    "Default": DefaultAgent()
}


# Routing logic based on detected emotions
def route_to_agent(emotion):
    """Route to the appropriate agent based on detected emotion.
    
    Args:
        emotion: The primary emotion detected in the user's message
        
    Returns:
        The key of the selected agent from the sub_agents dictionary
    """
    # Map emotions to appropriate therapeutic approaches
    
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
    
    # Special case for loneliness
    elif emotion == "loneliness":
        return "Psychoanalysis"
    
    # Default to Default agent for safety
    else:
        return "Default"