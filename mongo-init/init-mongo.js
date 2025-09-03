db.createCollection('companies', {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["_id", "name"],
      properties: {
        _id: { bsonType: "string", description: "Ticker symbol (primary key)" }, //ticker as id/primary key
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
        weburl: { bsonType: "string" }
      }
    }
  }
});

db.createCollection('earnings_reports', {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["ticker", "year", "quarter"],
      properties: {
        ticker: { bsonType: "string" },
        date: { bsonType: "string" },
        epsActual: { bsonType: ["int", "double"] },
        epsEstimate: { bsonType: ["int", "double"] },
        hour: { bsonType: "string" },
        quarter: { bsonType: ["int", "double"] },
        revenueActual: { bsonType: ["int", "double"] },
        revenueEstimate: { bsonType: ["int", "double"] },
        year: { bsonType: ["int", "double"] },
      }
    }
  }
});
db.earnings_reports.createIndex({ ticker: 1, year: 1, quarter: 1 }, { unique: true });

db.createCollection('sec_fillings', {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["ticker", "accessNumber"],
      properties: {
        ticker: { bsonType: "string" },
        accessNumber: { bsonType: "string" },
        cik: { bsonType: "string" },
        form: { bsonType: "string" },
        filedDate: { bsonType: "string" },
        acceptedDate: { bsonType: "string" },
        reportUrl: { bsonType: "string" },
        filingUrl: { bsonType: "string" },
      }
    }
  }
});
db.sec_fillings.createIndex({ ticker: 1, accessNumber: 1 }, { unique: true });

db.createCollection('news', {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["ticker", "id"],
      properties: {
        ticker: { bsonType: "string" },
        category: { bsonType: "string" },
        datetime: { bsonType: ["int", "long", "double"] },
        headline: { bsonType: "string" },
        id: { bsonType: ["int", "long", "double"] },
        image: { bsonType: "string" },
        related: { bsonType: "string" },
        source: { bsonType: "string" },
        summary: { bsonType: "string" },
        url: { bsonType: "string" },
      }
    }
  }
});
db.news.createIndex({ ticker: 1, id: 1 }, { unique: true });

db.createCollection('market_data', {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["ticker", "t"],
      properties: {
        ticker: { bsonType: "string" },
        c: { bsonType: "double" },
        d: { bsonType: "double" },
        dp: { bsonType: "double" },
        h: { bsonType: "double" },
        l: { bsonType: "double" },
        o: { bsonType: "double" },
        pc: { bsonType: "double" },
        t: { bsonType: "long" },
      }
    }
  }
});
db.market_data.createIndex({ ticker: 1, t: 1 }, { unique: true });

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