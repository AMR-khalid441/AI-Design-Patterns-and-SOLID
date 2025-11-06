# Overview

SRP states that a module should have one, and only one, reason to change.

In ML pipelines, violations often happen when a single class handles data IO, feature engineering, training, evaluation, logging, and persistence.

This module contrasts a monolithic approach vs. a small set of focused components.


