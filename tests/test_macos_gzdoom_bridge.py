import numpy as np

from fly_doom.doom.interface import DoomAction
from fly_doom.doom.macos_gzdoom_bridge import GZDoomWindow, KEYCODES, MacOSGZDoomBridge


class FakeQuartz:
    def __init__(self):
        self.events = []

    def CGEventCreateKeyboardEvent(self, _, keycode, down):
        return (keycode, down)

    def CGEventPostToPid(self, pid, event):
        self.events.append((pid, event))


def test_send_action_posts_key_down_and_up_to_gzdoom_pid():
    quartz = FakeQuartz()
    window = GZDoomWindow(10, 1234, "gzdoom", 800, 600)
    bridge = MacOSGZDoomBridge(quartz=quartz, window_finder=lambda: window)
    bridge.attach()
    bridge.send_action(DoomAction.TURN_RIGHT)
    assert quartz.events == [(1234, (KEYCODES[DoomAction.TURN_RIGHT], True)), (1234, (KEYCODES[DoomAction.TURN_RIGHT], False))]


def test_noop_does_not_post_input():
    quartz = FakeQuartz()
    window = GZDoomWindow(10, 1234, "gzdoom", 800, 600)
    bridge = MacOSGZDoomBridge(quartz=quartz, window_finder=lambda: window)
    bridge.attach()
    bridge.send_action(DoomAction.NOOP)
    assert quartz.events == []


def test_controller_handles_arbitrary_resolution_frames():
    from fly_doom.control.controllers import ControlledT4Controller
    from fly_doom.doom.interface import DoomObservation

    controller = ControlledT4Controller(width=64, height=64)
    # Native GZDoom high-resolution frame (e.g. 320x640)
    big_rgb = np.random.randint(0, 255, (320, 640, 3), dtype=np.uint8)
    obs = DoomObservation(rgb=big_rgb, depth=np.zeros((320, 640), dtype=np.float32))

    action = controller.select_action(obs)
    assert action in list(DoomAction)
