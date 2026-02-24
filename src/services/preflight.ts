// Copyright (C) 2025 Keygraph, Inc.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License version 3
// as published by the Free Software Foundation.

/**
 * Preflight Validation Service
 *
 * Runs cheap, fast checks before any agent execution begins.
 * Catches configuration and credential problems early, saving
 * time and API costs compared to failing mid-pipeline.
 *
 * Checks run sequentially, cheapest first:
 * 1. Repository path exists and contains .git
 * 2. Config file parses and validates (if provided)
 * 3. Credentials validate via OpenAI Chat Completions API
 */

import fs from 'fs/promises';
import { PentestError, isRetryableError } from './error-handling.js';
import { ErrorCode } from '../types/errors.js';
import { type Result, ok, err } from '../types/result.js';
import { parseConfig } from '../config-parser.js';
import type { ActivityLogger } from '../types/activity-logger.js';

// === Repository Validation ===

async function validateRepo(
  repoPath: string,
  logger: ActivityLogger
): Promise<Result<void, PentestError>> {
  logger.info('Checking repository path...', { repoPath });

  // 1. Check repo directory exists
  try {
    const stats = await fs.stat(repoPath);
    if (!stats.isDirectory()) {
      return err(
        new PentestError(
          `Repository path is not a directory: ${repoPath}`,
          'config',
          false,
          { repoPath },
          ErrorCode.REPO_NOT_FOUND
        )
      );
    }
  } catch {
    return err(
      new PentestError(
        `Repository path does not exist: ${repoPath}`,
        'config',
        false,
        { repoPath },
        ErrorCode.REPO_NOT_FOUND
      )
    );
  }

  // 2. Check .git directory exists
  try {
    const gitStats = await fs.stat(`${repoPath}/.git`);
    if (!gitStats.isDirectory()) {
      return err(
        new PentestError(
          `Not a git repository (no .git directory): ${repoPath}`,
          'config',
          false,
          { repoPath },
          ErrorCode.REPO_NOT_FOUND
        )
      );
    }
  } catch {
    return err(
      new PentestError(
        `Not a git repository (no .git directory): ${repoPath}`,
        'config',
        false,
        { repoPath },
        ErrorCode.REPO_NOT_FOUND
      )
    );
  }

  logger.info('Repository path OK');
  return ok(undefined);
}

// === Config Validation ===

async function validateConfig(
  configPath: string,
  logger: ActivityLogger
): Promise<Result<void, PentestError>> {
  logger.info('Validating configuration file...', { configPath });

  try {
    await parseConfig(configPath);
    logger.info('Configuration file OK');
    return ok(undefined);
  } catch (error) {
    if (error instanceof PentestError) {
      return err(error);
    }
    const message = error instanceof Error ? error.message : String(error);
    return err(
      new PentestError(
        `Configuration validation failed: ${message}`,
        'config',
        false,
        { configPath },
        ErrorCode.CONFIG_VALIDATION_FAILED
      )
    );
  }
}

// === Credential Validation ===

/** Validate credentials via a minimal ChatGPT API request. */
async function validateCredentials(
  logger: ActivityLogger
): Promise<Result<void, PentestError>> {
  // 1. Check that API key is present
  if (!process.env.OPENAI_API_KEY) {
    return err(
      new PentestError(
        'No API credentials found. Set OPENAI_API_KEY in .env',
        'config',
        false,
        {},
        ErrorCode.AUTH_FAILED
      )
    );
  }

  // 2. Validate via a minimal Chat Completions request
  const authType = 'OpenAI API key';
  logger.info(`Validating ${authType}...`);

  try {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: process.env.OPENAI_MODEL || 'gpt-4.1-mini',
        messages: [{ role: 'user', content: 'ping' }],
        max_completion_tokens: 16,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`OpenAI API error (${response.status}): ${errorText}`);
    }

    logger.info(`${authType} OK`);
    return ok(undefined);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const retryable = isRetryableError(error instanceof Error ? error : new Error(message));

    return err(
      new PentestError(
        retryable
          ? `Failed to reach OpenAI API. Check your network connection.`
          : `${authType} validation failed: ${message}`,
        retryable ? 'network' : 'config',
        retryable,
        { authType },
        retryable ? undefined : ErrorCode.AUTH_FAILED
      )
    );
  }
}

// === Preflight Orchestrator ===

/**
 * Run all preflight checks sequentially (cheapest first).
 *
 * 1. Repository path exists and contains .git
 * 2. Config file parses and validates (if configPath provided)
 * 3. Credentials validate (OpenAI API key)
 *
 * Returns on first failure.
 */
export async function runPreflightChecks(
  repoPath: string,
  configPath: string | undefined,
  logger: ActivityLogger
): Promise<Result<void, PentestError>> {
  // 1. Repository check (free — filesystem only)
  const repoResult = await validateRepo(repoPath, logger);
  if (!repoResult.ok) {
    return repoResult;
  }

  // 2. Config check (free — filesystem + CPU)
  if (configPath) {
    const configResult = await validateConfig(configPath, logger);
    if (!configResult.ok) {
      return configResult;
    }
  }

  // 3. Credential check (cheap — 1 API round-trip)
  const credResult = await validateCredentials(logger);
  if (!credResult.ok) {
    return credResult;
  }

  logger.info('All preflight checks passed');
  return ok(undefined);
}
