setupFile = FileNameJoin[{DirectoryName[$InputFileName], "mathematica_data_setup.wl"}];

Get[setupFile];

verification = VerifyBankForecastFiles[];
importSmokeTest = SmokeTestBankForecastImports[];

If[verification === $Failed,
  Exit[1],
  Print[InputForm[Normal[verification]]];
  Print[InputForm[Normal[importSmokeTest]]];
  Exit[0]
];
