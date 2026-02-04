"""
State Management - Agent state tracking
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    LISTENING = "listening"
    OBSERVING = "observing"
    ANALYZING = "analyzing"
    ACTING = "acting"
    AUTHENTICATING = "authenticating"
    VERIFYING = "verifying"
    RESPONDING = "responding"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class TaskState:
    """State for a single task"""
    task_id: str
    user_intent: str
    current_state: AgentState = AgentState.IDLE
    steps_completed: List[str] = field(default_factory=list)
    current_step: Optional[str] = None
    screen_context: Optional[Dict[str, Any]] = None
    detected_elements: List[Dict[str, Any]] = field(default_factory=list)
    error_count: int = 0
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStateManager:
    """Manages agent state across tasks"""
    
    current_task: Optional[TaskState] = None
    task_history: List[TaskState] = field(default_factory=list)
    authentication_state: Optional[Dict[str, Any]] = None
    device_info: Optional[Dict[str, Any]] = None
    last_screen_analysis: Optional[Dict[str, Any]] = None
    cached_screen: Optional[bytes] = None
    
    def start_task(self, task_id: str, user_intent: str) -> TaskState:
        """
        Start a new task
        
        Args:
            task_id: Unique task identifier
            user_intent: User's intent/command
            
        Returns:
            New TaskState
        """
        self.current_task = TaskState(
            task_id=task_id,
            user_intent=user_intent
        )
        return self.current_task
    
    def update_state(self, new_state: AgentState):
        """Update current task state"""
        if self.current_task:
            self.current_task.current_state = new_state
    
    def add_step(self, step: str):
        """Add completed step"""
        if self.current_task:
            self.current_task.steps_completed.append(step)
    
    def set_current_step(self, step: str):
        """Set current step"""
        if self.current_task:
            self.current_task.current_step = step
    
    def increment_error(self):
        """Increment error count"""
        if self.current_task:
            self.current_task.error_count += 1
    
    def increment_retry(self):
        """Increment retry count"""
        if self.current_task:
            self.current_task.retry_count += 1
    
    def complete_task(self):
        """Mark current task as completed"""
        if self.current_task:
            self.current_task.current_state = AgentState.COMPLETED
            self.task_history.append(self.current_task)
            self.current_task = None
    
    def set_authentication_state(self, auth_state: Dict[str, Any]):
        """Set authentication state"""
        self.authentication_state = auth_state
    
    def clear_authentication_state(self):
        """Clear authentication state"""
        self.authentication_state = None
    
    def update_screen_context(self, context: Dict[str, Any]):
        """Update screen context"""
        if self.current_task:
            self.current_task.screen_context = context
        self.last_screen_analysis = context
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get current state summary"""
        return {
            "current_task": {
                "task_id": self.current_task.task_id if self.current_task else None,
                "intent": self.current_task.user_intent if self.current_task else None,
                "state": self.current_task.current_state.value if self.current_task else None,
                "steps_completed": self.current_task.steps_completed if self.current_task else [],
                "error_count": self.current_task.error_count if self.current_task else 0,
            },
            "has_auth_state": self.authentication_state is not None,
            "task_count": len(self.task_history)
        }
