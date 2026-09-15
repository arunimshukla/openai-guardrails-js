/**
 * OpenAI Vector Store Creation Utility
 *
 * This module provides utilities for creating OpenAI vector stores from files or directories,
 * providing functionality for creating OpenAI vector stores.
 *
 * Note: This implementation uses OpenAI v4 API.
 */

import OpenAI from 'openai';

/**
 * Configuration for creating an OpenAI vector store.
 */
export interface OpenAIVectorStoreConfig {
  /** OpenAI API key */
  apiKey: string;
  /** Name for the vector store */
  name?: string;
}

/**
 * Create an OpenAI vector store from files or directories.
 *
 * This function creates a vector store by:
 * 1. Creating an assistant with file search capabilities
 * 2. Uploading files to OpenAI
 * 3. Attaching files to the assistant
 *
 * @param path - Path to file or directory containing documents
 * @param config - Configuration for the OpenAI client
 * @returns Assistant ID that can be used as a knowledge source
 */
export async function createOpenAIVectorStoreFromPath(
  path: string,
  config: OpenAIVectorStoreConfig
): Promise<string> {
  const client = new OpenAI({ apiKey: config.apiKey });

  // Check if path exists
  try {
    const fs = await import('node:fs/promises');
    await fs.access(path);
  } catch {
    throw new Error(`Path does not exist: ${path}`);
  }

  try {
    // Get list of files to upload
    const filePaths = await getFilePaths(path);

    if (filePaths.length === 0) {
      throw new Error(`No supported files found in ${path}`);
    }

    // Upload files
    const fileIds = await uploadFiles(client, filePaths);

    if (fileIds.length === 0) {
      throw new Error('No files were successfully uploaded');
    }

    // Create a vector store
    const vectorStore = await client.vectorStores.create({
      name: config.name || `anti_hallucination_${path.split('/').pop() || 'documents'}`,
    });

    // Attach files to the vector store
    for (const fileId of fileIds) {
      await client.vectorStores.files.create(vectorStore.id, { file_id: fileId });
    }

    // Wait for vector-store indexing, not global uploaded-file processing
    await waitForFileProcessing(client, vectorStore.id, fileIds);

    // Return the vector store ID
    return vectorStore.id;
  } catch (error) {
    throw new Error(
      `Failed to create vector store: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}

/**
 * Get list of supported files from a path.
 */
async function getFilePaths(path: string): Promise<string[]> {
  const fs = await import('node:fs/promises');
  const pathModule = await import('node:path');

  const supportedFileTypes = [
    '.c',
    '.cpp',
    '.cs',
    '.css',
    '.doc',
    '.docx',
    '.go',
    '.html',
    '.java',
    '.js',
    '.json',
    '.md',
    '.pdf',
    '.php',
    '.pptx',
    '.py',
    '.rb',
    '.sh',
    '.tex',
    '.ts',
    '.txt',
  ];

  // Check extension before stat if it looks like a file
  const ext = pathModule.extname(path).toLowerCase();
  if (ext && !supportedFileTypes.includes(ext)) {
    // If the path has an extension and it's not supported, skip stat and return []
    return [];
  }

  try {
    const stat = await fs.stat(path);

    if (stat.isFile()) {
      // ext already calculated above
      return supportedFileTypes.includes(ext) ? [path] : [];
    } else if (stat.isDirectory()) {
      const files: string[] = [];
      const entries = await fs.readdir(path, { withFileTypes: true });

      for (const entry of entries) {
        if (entry.isFile()) {
          const fullPath = pathModule.join(path, entry.name);
          const entryExt = pathModule.extname(entry.name).toLowerCase();
          if (supportedFileTypes.includes(entryExt)) {
            files.push(fullPath);
          }
        }
      }

      return files;
    }
  } catch (error) {
    throw new Error(
      `Error reading path ${path}: ${error instanceof Error ? error.message : String(error)}`
    );
  }

  return [];
}

/**
 * Upload files to OpenAI and return file IDs.
 */
async function uploadFiles(client: OpenAI, filePaths: string[]): Promise<string[]> {
  const fs = await import('node:fs/promises');
  const fileIds: string[] = [];

  for (const filePath of filePaths) {
    try {
      const fileBuffer = await fs.readFile(filePath);
      const pathModule = await import('node:path');
      const fileName = pathModule.basename(filePath);

      // Create a File-like object that matches the Uploadable interface
      const file = await client.files.create({
        file: new File([fileBuffer], fileName, { type: 'application/octet-stream' }),
        purpose: 'assistants',
      });
      fileIds.push(file.id);
    } catch (error) {
      console.warn(
        `Failed to upload file ${filePath}: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  return fileIds;
}

/**
 * Wait for attached files to finish indexing in the vector store.
 */
async function waitForFileProcessing(
  client: OpenAI,
  vectorStoreId: string,
  fileIds: string[]
): Promise<void> {
  while (true) {
    const allCompleted = await Promise.all(
      fileIds.map(async (fileId) => {
        const file = await client.vectorStores.files.retrieve(fileId, {
          vector_store_id: vectorStoreId,
        });
        if (file.status === 'failed' || file.status === 'cancelled') {
          const detail = file.last_error?.message;
          throw new Error(
            `File ${fileId} did not complete indexing in vector store ${vectorStoreId}: ${file.status}${detail ? ` (${detail})` : ''}`
          );
        }
        return file.status === 'completed';
      })
    );

    if (allCompleted.every((status) => status)) {
      return;
    }

    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}
