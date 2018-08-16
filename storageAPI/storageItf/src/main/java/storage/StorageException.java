package storage;

import java.util.*;
import java.lang.*;

import java.io.*;
import static java.lang.System.out;

public class StorageException extends Exception{

    public StorageException(String message) {
        super(message);
    }

    public StorageException(String message, Throwable throwable) {
        super(message, throwable);
    }
    
    public String getMessage()
    {
        return super.getMessage();
    }
}