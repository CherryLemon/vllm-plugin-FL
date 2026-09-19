# SPDX-License-Identifier: Apache-2.0
"""Bind model adapters through the common resolver without hiding GPU failures."""

from .policy import get_policy, get_policy_epoch


class OperatorBinding:
    """A cached common-dispatch binding, refreshed when public policy changes.

    Only NotImplementedError denotes an unsupported workload. OOM, launch,
    numerical and programming failures propagate unchanged. A rejected
    implementation is disabled for this manager instead of retried per token.
    """

    def __init__(self, manager, op_name, *, graph_capabilities=None):
        self.manager = manager
        self.op_name = op_name
        self.graph_capabilities = graph_capabilities or {}
        self._epoch = None
        self._candidates = []
        self.selected_impl = None

    def preflight(self):
        epoch = get_policy_epoch()
        if epoch != self._epoch:
            self._candidates = self.manager.resolve_candidates(self.op_name)
            self._strict = get_policy().strict
            failed = self.manager.get_failed_impls(self.op_name).get(
                self.op_name, set()
            )
            self._candidates = [
                impl for impl in self._candidates if impl.impl_id not in failed
            ]
            self._epoch = epoch
        if not self._candidates:
            raise RuntimeError(
                f"No permitted implementation remains for {self.op_name}"
            )
        return self._candidates

    def describe(self):
        candidates = self.preflight()
        return dict(
            op=self.op_name,
            selected=self.selected_impl or candidates[0].impl_id,
            candidates=[impl.impl_id for impl in candidates],
            strict=self._strict,
            fallback_on="NotImplementedError only" if not self._strict else "never",
            graph_capabilities=self.graph_capabilities,
        )

    def __call__(self, *args, **kwargs):
        candidates = self.preflight()
        for impl in candidates:
            self.manager._record_first_use(self.op_name, impl)
            try:
                result = impl.fn(*args, **kwargs)
            except NotImplementedError:
                if self._strict:
                    raise
                self.manager._mark_failed_impl(self.op_name, impl.impl_id)
                self._candidates = [c for c in self._candidates if c is not impl]
                if impl is candidates[-1]:
                    raise
            else:
                self.selected_impl = impl.impl_id
                return result
