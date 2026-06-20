from streamlit.testing.v1 import AppTest


def test_all_dashboard_workspaces_render_without_exceptions():
    pages = [
        "Command Center",
        "Türkiye Focus",
        "Blind Tournament Audit",
        "Model Evolution",
        "Advanced Laboratory",
        "Tournament Engine",
        "Match Laboratory",
        "Model Observatory",
        "Team Atlas",
        "Research Artifacts",
    ]
    app = AppTest.from_file("app.py", default_timeout=120)
    app.run()
    for page in pages:
        app.radio[0].set_value(page).run()
        assert not app.exception, page
