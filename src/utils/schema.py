@schema
class SessionTask(dj.Imported):
    definition: """
    # Establish session-task association
    -> Session
    ---
    -> Task
    """

@schema
class DynamicTraining(dj.Imported):
    definition = """
    -> SessionTask
    """
    key_source = SessionTask & 'task_name = "dynamic_training"'

    class Block(dj.Part):
        definition = """
        -> master
        block_num: int  # unique block number within session
        ---
        current_rule: varchar(12)   # rule of block
        """

    class Trial(dj.Part):
        definition = """
        # Trials in Dynamic Training. All times are relative to session clock in secs
        -> master
        trial_num: int  # unique trial number
        ---
        -> master.Block
        trial_start: unsigned int # trial start time
        fixation_onset: float   # fixation onset time   
        stimulus_onset: float   # stimulus onset time
        valid: bool     # if trial is invalid (false) or valid (true)
        correct: bool   # if the trial is correct or not
        response: ENUM('Correct', 'Incorrect', 'NoResponse')
        response_time: float
        timeout_

        """

    class TrialCondition(dj.Part):
        definition = """
        -> master.Trial
        coherence: float    # coherence of current trial
        ---
        """
    
    