(* Bank transaction volume forecast: local Mathematica data setup.
   Load this from notebooks/main_mathematica.nb with:
   Get[FileNameJoin[{NotebookDirectory[], "mathematica_data_setup.wl"}]]
*)

ClearAll[
  $BankForecastProjectRoot,
  $BankForecastInputDir,
  $BankForecastOutputDir,
  $BankForecastFiles,
  VerifyBankForecastFiles,
  ImportBankForecastCSV,
  ImportBankForecastParquet,
  ImportBankForecastFile,
  LoadBankForecastData,
  SmokeTestBankForecastImports
];

If[! ValueQ[$BankForecastProjectRoot],
  $BankForecastProjectRoot = "H:\\hikrepos\\bank-transaction-volume-forecast";
];

$BankForecastInputDir = FileNameJoin[{$BankForecastProjectRoot, "data", "inputs"}];
$BankForecastOutputDir = FileNameJoin[{$BankForecastProjectRoot, "outputs"}];

$BankForecastFiles = <|
  "Transactions" -> <|
    "Path" -> FileNameJoin[{$BankForecastInputDir, "transactions_features.parquet"}],
    "Required" -> True,
    "Format" -> "Parquet"
  |>,
  "Financials" -> <|
    "Path" -> FileNameJoin[{$BankForecastInputDir, "financials_features.parquet"}],
    "Required" -> True,
    "Format" -> "Parquet"
  |>,
  "Demographics" -> <|
    "Path" -> FileNameJoin[{$BankForecastInputDir, "demographics_clean.parquet"}],
    "Required" -> True,
    "Format" -> "Parquet"
  |>,
  "TrainLabels" -> <|
    "Path" -> FileNameJoin[{$BankForecastInputDir, "Train.csv"}],
    "Required" -> True,
    "Format" -> "CSV"
  |>,
  "TestCustomers" -> <|
    "Path" -> FileNameJoin[{$BankForecastInputDir, "Test.csv"}],
    "Required" -> True,
    "Format" -> "CSV"
  |>,
  "VariableDefinitions" -> <|
    "Path" -> FileNameJoin[{$BankForecastInputDir, "VariableDefinitions.csv"}],
    "Required" -> True,
    "Format" -> "CSV"
  |>,
  "SampleSubmission" -> <|
    "Path" -> FileNameJoin[{$BankForecastInputDir, "SampleSubmission.csv"}],
    "Required" -> False,
    "Format" -> "CSV"
  |>
|>;

VerifyBankForecastFiles[] := Module[{rows, requiredMissing},
  rows = KeyValueMap[
    <|
      "Name" -> #1,
      "Path" -> #2["Path"],
      "Required" -> #2["Required"],
      "Format" -> #2["Format"],
      "Exists" -> FileExistsQ[#2["Path"]],
      "SizeMB" -> If[
        FileExistsQ[#2["Path"]],
        NumberForm[N[FileByteCount[#2["Path"]]/1024.^2], {8, 2}],
        Missing["NotAvailable"]
      ]
    |>&,
    $BankForecastFiles
  ];

  requiredMissing = Select[rows, TrueQ[#["Required"]] && ! TrueQ[#["Exists"]]&];

  If[Length[requiredMissing] > 0,
    Print["Missing required project files:"];
    Print[Dataset[requiredMissing]];
    $Failed,
    Dataset[rows]
  ]
];

ImportBankForecastCSV[file_String] := Module[{rows, header, records},
  If[! FileExistsQ[file],
    Return[Failure["FileMissing", <|"Path" -> file|>]]
  ];

  rows = Import[file, "CSV"];
  If[! ListQ[rows] || Length[rows] == 0,
    Return[Dataset[{}]]
  ];

  header = ToString /@ First[rows];
  records = AssociationThread[header, #]& /@ Rest[rows];
  Dataset[records]
];

ImportBankForecastParquet[file_String] := Module[{directImport, pythonCode, pythonImport},
  If[! FileExistsQ[file],
    Return[Failure["FileMissing", <|"Path" -> file|>]]
  ];

  directImport = Quiet[Check[Import[file, "Dataset"], $Failed]];
  If[directImport =!= $Failed,
    Return[directImport]
  ];

  pythonCode = StringTemplate[
    "import pandas as pd\n" <>
    "df = pd.read_parquet(r'''`path`''')\n" <>
    "df.head(`limit`).to_dict(orient='records')"
  ][<|"path" -> file, "limit" -> 200000|>];

  pythonImport = Quiet[Check[ExternalEvaluate["Python", pythonCode], $Failed]];
  If[pythonImport === $Failed,
    Failure[
      "ParquetImportUnavailable",
      <|
        "Path" -> file,
        "Message" -> "Mathematica could not import Parquet directly and Python ExternalEvaluate failed. Export a CSV sample from Python or configure Python for ExternalEvaluate."
      |>
    ],
    Dataset[pythonImport]
  ]
];

ImportBankForecastFile[name_String] := Module[{fileInfo},
  If[! KeyExistsQ[$BankForecastFiles, name],
    Return[Failure["UnknownInputName", <|"Name" -> name|>]]
  ];

  fileInfo = $BankForecastFiles[name];
  Switch[fileInfo["Format"],
    "CSV", ImportBankForecastCSV[fileInfo["Path"]],
    "Parquet", ImportBankForecastParquet[fileInfo["Path"]],
    _, Failure["UnsupportedFormat", fileInfo]
  ]
];

LoadBankForecastData[names_: Automatic] := Module[{selectedNames, verification},
  verification = VerifyBankForecastFiles[];
  If[verification === $Failed, Return[$Failed]];

  selectedNames = If[names === Automatic,
    Keys @ Select[$BankForecastFiles, TrueQ[#["Required"]]&],
    names
  ];

  AssociationMap[ImportBankForecastFile, selectedNames]
];

SmokeTestBankForecastImports[] := Module[{verification, trainLabels, demographics},
  verification = VerifyBankForecastFiles[];
  If[verification === $Failed, Return[$Failed]];

  trainLabels = ImportBankForecastFile["TrainLabels"];
  demographics = ImportBankForecastFile["Demographics"];

  If[FailureQ[trainLabels] || FailureQ[demographics],
    Return[
      Failure[
        "ImportSmokeTestFailed",
        <|
          "TrainLabels" -> trainLabels,
          "Demographics" -> demographics
        |>
      ]
    ]
  ];

  Dataset[{
    <|
      "Table" -> "TrainLabels",
      "RowsLoaded" -> Length[Normal[trainLabels]],
      "ImportStatus" -> "OK"
    |>,
    <|
      "Table" -> "Demographics",
      "RowsLoaded" -> Length[Normal[demographics]],
      "ImportStatus" -> "OK"
    |>
  }]
];

Print["Bank forecast Mathematica setup loaded."];
Print["Project root: ", $BankForecastProjectRoot];
Print["Run VerifyBankForecastFiles[] to confirm local data access."];
