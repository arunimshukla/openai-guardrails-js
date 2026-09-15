/**
 * Tests for OpenAI vector store creation utilities.
 */

import type { VectorStoreFile } from 'openai/resources/vector-stores/files';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const fsMock = {
  access: vi.fn(),
  stat: vi.fn(),
  readdir: vi.fn(),
  readFile: vi.fn(),
};

const openAiState = {
  failUploads: false,
  uploadCount: 0,
};

const completedVectorStoreFile: VectorStoreFile = {
  id: 'file_1',
  created_at: 0,
  last_error: null,
  object: 'vector_store.file',
  status: 'completed',
  usage_bytes: 0,
  vector_store_id: 'vs_123',
};

const vectorStoreFilesMock = {
  retrieve:
    vi.fn<(fileId: string, params: { vector_store_id: string }) => Promise<VectorStoreFile>>(),
};

const openAiInstances: MockOpenAI[] = [];

class MockOpenAI {
  public files = {
    create: vi.fn(async () => {
      if (openAiState.failUploads) {
        throw new Error('upload failure');
      }
      openAiState.uploadCount += 1;
      return { id: `file_${openAiState.uploadCount}` };
    }),
    retrieve: vi.fn(async () => ({ status: 'processed' })),
  };

  public vectorStores = {
    create: vi.fn(async () => ({ id: 'vs_123' })),
    files: {
      create: vi.fn(async () => completedVectorStoreFile),
      retrieve: vectorStoreFilesMock.retrieve,
    },
  };

  constructor(public config: { apiKey: string }) {
    // no-op
    openAiInstances.push(this);
  }
}

vi.mock('fs/promises', () => fsMock);
vi.mock('openai', () => ({
  default: MockOpenAI,
}));

describe('createOpenAIVectorStoreFromPath', () => {
  let createOpenAIVectorStoreFromPath: (
    path: string,
    config: { apiKey: string }
  ) => Promise<string>;

  beforeEach(async () => {
    openAiState.failUploads = false;
    openAiState.uploadCount = 0;
    openAiInstances.length = 0;
    vectorStoreFilesMock.retrieve.mockReset();
    vectorStoreFilesMock.retrieve.mockImplementation(async (fileId) => ({
      ...completedVectorStoreFile,
      id: fileId,
    }));
    fsMock.access.mockReset();
    fsMock.stat.mockReset();
    fsMock.readdir.mockReset();
    fsMock.readFile.mockReset();
    global.File =
      global.File ||
      (class PolyfillFile {
        constructor(
          public blobs: unknown[],
          public name: string,
          public options: Record<string, unknown>
        ) {}
      } as unknown as typeof File);

    vi.resetModules();
    ({ createOpenAIVectorStoreFromPath } = await import('../../../utils/openai-vector-store'));
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.resetModules();
  });

  it('creates a vector store from directory files', async () => {
    fsMock.access.mockResolvedValue(undefined);
    fsMock.stat.mockResolvedValue({
      isFile: () => false,
      isDirectory: () => true,
    });
    fsMock.readdir.mockResolvedValue([
      {
        isFile: () => true,
        name: 'doc.txt',
      },
    ]);
    fsMock.readFile.mockResolvedValue(new Uint8Array([1, 2, 3]));

    const id = await createOpenAIVectorStoreFromPath('/tmp/docs', { apiKey: 'k' });

    expect(id).toBe('vs_123');
    expect(openAiInstances[0].vectorStores.create).toHaveBeenCalled();
    expect(openAiInstances[0].files.create).toHaveBeenCalledWith(
      expect.objectContaining({
        file: expect.any(File),
        purpose: 'assistants',
      })
    );
    expect(openAiInstances[0].vectorStores.files.create).toHaveBeenCalledWith('vs_123', {
      file_id: 'file_1',
    });
    expect(vectorStoreFilesMock.retrieve).toHaveBeenCalledWith('file_1', {
      vector_store_id: 'vs_123',
    });
    expect(openAiInstances[0].files.retrieve).not.toHaveBeenCalled();
  });

  it('waits for every vector-store file to complete indexing before returning', async () => {
    vi.useFakeTimers();
    fsMock.access.mockResolvedValue(undefined);
    fsMock.stat.mockResolvedValue({
      isFile: () => false,
      isDirectory: () => true,
    });
    fsMock.readdir.mockResolvedValue([
      { isFile: () => true, name: 'first.txt' },
      { isFile: () => true, name: 'second.txt' },
    ]);
    fsMock.readFile.mockResolvedValue(new Uint8Array([1, 2, 3]));
    vectorStoreFilesMock.retrieve
      .mockResolvedValueOnce(completedVectorStoreFile)
      .mockResolvedValueOnce({
        ...completedVectorStoreFile,
        id: 'file_2',
        status: 'in_progress',
      });
    const settled = vi.fn();
    const result = createOpenAIVectorStoreFromPath('/tmp/docs', { apiKey: 'k' });
    void result.then(settled, settled);

    await vi.waitFor(() => {
      expect(vectorStoreFilesMock.retrieve).toHaveBeenCalledTimes(2);
    });
    expect(openAiInstances[0].vectorStores.files.create).toHaveBeenNthCalledWith(1, 'vs_123', {
      file_id: 'file_1',
    });
    expect(openAiInstances[0].vectorStores.files.create).toHaveBeenNthCalledWith(2, 'vs_123', {
      file_id: 'file_2',
    });
    expect(vectorStoreFilesMock.retrieve).toHaveBeenNthCalledWith(1, 'file_1', {
      vector_store_id: 'vs_123',
    });
    expect(vectorStoreFilesMock.retrieve).toHaveBeenNthCalledWith(2, 'file_2', {
      vector_store_id: 'vs_123',
    });
    expect(settled).not.toHaveBeenCalled();
    expect(openAiInstances[0].files.retrieve).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(1000);
    await expect(result).resolves.toBe('vs_123');
    expect(vectorStoreFilesMock.retrieve).toHaveBeenCalledTimes(4);
  });

  it.each(['failed', 'cancelled'] as const)(
    'rejects when vector-store indexing is %s',
    async (status) => {
      fsMock.access.mockResolvedValue(undefined);
      fsMock.stat.mockResolvedValue({
        isFile: () => false,
        isDirectory: () => true,
      });
      fsMock.readdir.mockResolvedValue([{ isFile: () => true, name: 'doc.txt' }]);
      fsMock.readFile.mockResolvedValue(new Uint8Array([1, 2, 3]));
      vectorStoreFilesMock.retrieve.mockResolvedValue({
        ...completedVectorStoreFile,
        status,
      });

      await expect(createOpenAIVectorStoreFromPath('/tmp/docs', { apiKey: 'k' })).rejects.toThrow(
        `File file_1 did not complete indexing in vector store vs_123: ${status}`
      );
      expect(openAiInstances[0].files.retrieve).not.toHaveBeenCalled();
    }
  );

  it('includes the indexing error reported by the API', async () => {
    fsMock.access.mockResolvedValue(undefined);
    fsMock.stat.mockResolvedValue({
      isFile: () => true,
      isDirectory: () => false,
    });
    fsMock.readFile.mockResolvedValue(new Uint8Array([1, 2, 3]));
    vectorStoreFilesMock.retrieve.mockResolvedValue({
      ...completedVectorStoreFile,
      status: 'failed',
      last_error: { code: 'invalid_file', message: 'The document could not be parsed' },
    });

    await expect(createOpenAIVectorStoreFromPath('/tmp/doc.txt', { apiKey: 'k' })).rejects.toThrow(
      'File file_1 did not complete indexing in vector store vs_123: failed (The document could not be parsed)'
    );
  });

  it('propagates errors from vector-store polling', async () => {
    fsMock.access.mockResolvedValue(undefined);
    fsMock.stat.mockResolvedValue({
      isFile: () => true,
      isDirectory: () => false,
    });
    fsMock.readFile.mockResolvedValue(new Uint8Array([1, 2, 3]));
    vectorStoreFilesMock.retrieve.mockRejectedValue(new Error('polling failed'));

    await expect(createOpenAIVectorStoreFromPath('/tmp/doc.txt', { apiKey: 'k' })).rejects.toThrow(
      'Failed to create vector store: polling failed'
    );
  });

  it('throws when directory has no supported files', async () => {
    fsMock.access.mockResolvedValue(undefined);
    fsMock.stat.mockResolvedValue({
      isFile: () => false,
      isDirectory: () => true,
    });
    fsMock.readdir.mockResolvedValue([]);

    await expect(createOpenAIVectorStoreFromPath('/tmp/docs', { apiKey: 'key' })).rejects.toThrow(
      'No supported files found in /tmp/docs'
    );
  });

  it('throws when uploads fail and no files were uploaded', async () => {
    openAiState.failUploads = true;
    fsMock.access.mockResolvedValue(undefined);
    fsMock.stat.mockResolvedValue({
      isFile: () => false,
      isDirectory: () => true,
    });
    fsMock.readdir.mockResolvedValue([
      {
        isFile: () => true,
        name: 'doc.txt',
      },
    ]);
    fsMock.readFile.mockResolvedValue(new Uint8Array([1, 2, 3]));

    await expect(createOpenAIVectorStoreFromPath('/tmp/docs', { apiKey: 'key' })).rejects.toThrow(
      'No files were successfully uploaded'
    );
  });
});
