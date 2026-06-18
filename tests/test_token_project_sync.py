import tempfile
import unittest
from datetime import datetime, timezone

from src.core.config import config
from src.core.database import Database
from src.core.models import Project, Token
from src.services.token_manager import TokenManager


class _DummyFlowClient:
    async def create_project(self, st: str, project_name: str) -> str:
        raise AssertionError("create_project should not be called in this test")


class TokenProjectSyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(db_path=f"{self._temp_dir.name}/flow.db")
        await self.db.init_db()
        self._original_pool_size = config.personal_project_pool_size
        config.set_personal_project_pool_size(1)
        self.manager = TokenManager(self.db, _DummyFlowClient())

    async def asyncTearDown(self):
        config.set_personal_project_pool_size(self._original_pool_size)
        self._temp_dir.cleanup()

    async def _create_token(self) -> int:
        token = Token(
            st="st-test",
            at="at-test",
            at_expires=datetime.now(timezone.utc),
            email="user@example.com",
            name="user",
            current_project_id="old-project",
            current_project_name="Old Project P1",
        )
        token_id = await self.db.add_token(token)
        await self.db.add_project(
            Project(
                project_id="old-project",
                token_id=token_id,
                project_name="Old Project P1",
                tool_name="PINHOLE",
                is_active=True,
            )
        )
        return token_id

    async def test_update_token_replaces_single_project_pool(self):
        token_id = await self._create_token()

        await self.manager.update_token(
            token_id=token_id,
            project_id="new-project",
            project_name="New Project P1",
        )

        token = await self.db.get_token(token_id)
        projects = await self.db.get_projects_by_token(token_id)

        self.assertEqual(token.current_project_id, "new-project")
        self.assertEqual(len(projects), 1)
        self.assertEqual(projects[0].project_id, "new-project")
        self.assertEqual(projects[0].project_name, "New Project P1")

    async def test_ensure_project_exists_self_heals_stale_single_project_pool(self):
        token_id = await self._create_token()
        await self.db.update_token(
            token_id,
            current_project_id="manual-project",
            current_project_name="Manual Project P1",
        )

        project_id = await self.manager.ensure_project_exists(token_id)
        projects = await self.db.get_projects_by_token(token_id)

        self.assertEqual(project_id, "manual-project")
        self.assertEqual(len(projects), 1)
        self.assertEqual(projects[0].project_id, "manual-project")
        self.assertEqual(projects[0].project_name, "Manual Project P1")


if __name__ == "__main__":
    unittest.main()
