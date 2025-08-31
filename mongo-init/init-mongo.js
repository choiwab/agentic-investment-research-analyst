db.createCollection('companies', {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      properties: {
        country: { bsonType: "string" },
        currency: { bsonType: "string" },
        exchange: { bsonType: "string" },
        finnhubIndustry: { bsonType: "string" },
        ipo: { bsonType: "string" },
        logo: { bsonType: "string" },
        marketCapitalization: { bsonType: "double" },
        name: { bsonType: "string" },
        phone: { bsonType: "string" },
        shareOutstanding: { bsonType: "double" },
        ticker: { bsonType: "string" },
        weburl: { bsonType: "string" }
      }
    }
  }
});
db.companies.createIndex({ ticker: 1 });

db.createCollection('earnings_reports', {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      properties: {
        date: { bsonType: "string" },
        epsActual: { bsonType: "double" },
        epsEstimate: { bsonType: "double" },
        hour: { bsonType: "string" },
        quarter: { bsonType: "double" },
        revenueActual: { bsonType: "double" },
        revenueEstimate: { bsonType: "double" },
        symbol: { bsonType: "string" },
        year: { bsonType: "double" },
        ticker: { bsonType: "string" }
      }
    }
  }
});
db.earnings_reports.createIndex({ ticker: 1 });

db.createCollection('sec_fillings', {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      properties: {
        accessNumber: { bsonType: "string" },
        symbol: { bsonType: "string" },
        cik: { bsonType: "string" },
        form: { bsonType: "string" },
        filedDate: { bsonType: "string" },
        acceptedDate: { bsonType: "string" },
        reportUrl: { bsonType: "string" },
        filingUrl: { bsonType: "string" },
        ticker: { bsonType: "string" }
      }
    }
  }
});
db.sec_fillings.createIndex({ ticker: 1 });

db.createCollection('news', {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      properties: {
        category: { bsonType: "string" },
        datetime: { bsonType: "long" },
        headline: { bsonType: "string" },
        id: { bsonType: "long" },
        image: { bsonType: "string" },
        related: { bsonType: "string" },
        source: { bsonType: "string" },
        summary: { bsonType: "string" },
        url: { bsonType: "string" },
        ticker: { bsonType: "string" }
      }
    }
  }
});
db.news.createIndex({ ticker: 1 });

db.createCollection('market_data', {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      properties: {
        c: { bsonType: "double" },
        d: { bsonType: "double" },
        dp: { bsonType: "double" },
        h: { bsonType: "double" },
        l: { bsonType: "double" },
        o: { bsonType: "double" },
        pc: { bsonType: "double" },
        t: { bsonType: "long" },
        ticker: { bsonType: "string" }
      }
    }
  }
});
db.market_data.createIndex({ ticker: 1 });

// db.createCollection('market_status', {
//   validator: {
//     $jsonSchema: {
//       bsonType: "object",
//       required: ["exchange", "isOpen", "t", "timezone"],
//       properties: {
//         exchange: {
//           bsonType: "string",
//           description: "must be a string and is required"
//         },
//         holiday: {
//           bsonType: ["string", "null"],
//           description: "must be a string or null"
//         },
//         isOpen: {
//           bsonType: "bool",
//           description: "must be a boolean and is required"
//         },
//         session: {
//           bsonType: ["string", "null"],
//           description: "must be a string or null"
//         },
//         t: {
//           bsonType: "long",
//           description: "must be a long and is required"
//         },
//         timezone: {
//           bsonType: "string",
//           description: "must be a string and is required"
//         }
//       }
//     }
//   }
// });
// db.market_status.createIndex({ exchange: 1 });