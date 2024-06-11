import logging
from typing import Optional, Dict, List
from uuid import uuid4, UUID

RunID = str | UUID

class Run:
  def __init__(self, run_id: str | None = None, parent_run_id: str | None = None):
    self.id: str = run_id or str(uuid4())
    self.parent_run_id: str | None = parent_run_id
    self.children: List[Run] = []

class RunManager:
  def __init__(self):
    self.runs: Dict[str, Run] = {}

  def start_run(self, run_id: RunID | None = None, parent_run_id: RunID | None = None) -> Run | None:
    if run_id != None and run_id == parent_run_id:
      logging.error("A run cannot be its own parent.")
      return None

    if isinstance(run_id, UUID):
      run_id = str(run_id)
    if isinstance(parent_run_id, UUID):
      parent_run_id = str(parent_run_id)
    if not self._run_exists(parent_run_id):
      # in Langchain CallbackHandler, sometimes it pass a parent_run_id for run that do not exist. Those runs should be ignored by Lunary
      parent_run_id = None
    
    run = Run(run_id, parent_run_id)
    self.runs[run.id] = run
    if parent_run_id:
      parent_run = self.runs.get(parent_run_id)
      if parent_run:
        parent_run.children.append(run)
    return run

  def end_run(self, run_id: RunID) -> str:
    if isinstance(run_id, UUID):
      run_id = str(run_id)

    run = self.runs.get(run_id)
    if run:
      self._delete_run(run)
    
    return run_id

  def _run_exists(self, run_id: str) -> bool:
      return run_id in self.runs

  def _delete_run(self, run: Run) -> None:
    for child in run.children:
      self._delete_run(child)

    if run.parent_run_id:
      parent_run = self.runs.get(run.parent_run_id)
      if parent_run:
        parent_run.children.remove(run)

    if self.runs.get(run.id):
      del self.runs[run.id]
